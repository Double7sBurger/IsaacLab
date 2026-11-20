# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference tensors and differentiable kinematics in the algorithm's WXYZ convention."""

from pathlib import Path

import mujoco
import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .analytic_ik import AnalyticIK
from .kinematics import G1_JOINT_NAMES, TRACKED_BODIES, ArticulationKinematics
from .math import matrix_to_quat
from .motion import MotionLib
from .quaternion import from_native, to_native


class ForwardKinematics:
    """Adapt native articulation kinematics to the reference algorithm's callable interface."""

    def __init__(self, path: str, joint_names: list[str], body_names: list[str], device: str):
        self.kinematics = ArticulationKinematics(path, device, joint_names, body_names)
        self.path = path
        self.joint_names = joint_names
        self.body_names = body_names
        self.device = device

    def __call__(self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel):
        pos, rot, vel, ang_vel = self.kinematics.forward(
            root_pos, matrix_from_quat(to_native(root_quat)), root_lin_vel, root_ang_vel, joint_pos, joint_vel
        )
        return pos, matrix_to_quat(rot), vel, ang_vel

    def analytic_ik(self) -> AnalyticIK:
        """Build the reference maximal-coordinate joint reconstruction for this articulation."""
        model = mujoco.MjModel.from_xml_path(self.path)
        body_ids = torch.tensor([model.body(name).id for name in self.body_names])
        joint_q_adr = torch.tensor([model.jnt_qposadr[model.joint(name).id] for name in self.joint_names])
        return AnalyticIK(model, body_ids, joint_q_adr, torch.arange(len(self.body_names)), self.device, torch.float32)


class MotionReference:
    """Concatenated clips with exclusive ends, retaining supplied reference velocities.

    NPZ clips contain ``joint_pos``, ``joint_vel``, ``root_pos_w``, ``root_quat_w``
    (WXYZ), ``root_lin_vel_w``, and ``root_ang_vel_w`` at 50 Hz. CSV/NPY inputs use
    the existing XYZW pose preprocessing; their inferred velocities are not a
    substitute for the reference project's processed motion dataset.
    """

    def __init__(self, cfg, device: str):
        names = G1_JOINT_NAMES if cfg.joint_names is None else cfg.joint_names
        bodies = TRACKED_BODIES if cfg.tracked_bodies is None else cfg.tracked_bodies
        self.fk = ForwardKinematics(cfg.kinematics_path, names, bodies, device)
        self.device = device
        files = sorted(Path(cfg.motion_dir).rglob("*.npz"))
        shard_count = getattr(cfg, "motion_shard_count", 1)
        shard_index = getattr(cfg, "motion_shard_index", 0)
        if not 0 <= shard_index < shard_count:
            raise ValueError("Motion shard index must be in [0, motion_shard_count).")
        if shard_count > 1:
            files = files[shard_index::shard_count]
            if not files:
                raise ValueError("Sharded references require at least one NPZ clip per rank.")
        fields = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")
        if files:
            clips = []
            for path in files:
                with np.load(path, allow_pickle=False) as data:
                    clip = {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in fields}
                    if "fps" in data and float(data["fps"]) != 50.0:
                        raise ValueError(f"Reference clip {path} must be sampled at 50 Hz.")
                length = len(clip["joint_pos"])
                widths = (len(names), len(names), 3, 4, 3, 3)
                if length < 2 or any(clip[key].shape != (length, width) for key, width in zip(fields, widths)):
                    raise ValueError(f"Invalid reference layout in {path}.")
                if any(not torch.isfinite(value).all() for value in clip.values()):
                    raise ValueError(f"Nonfinite reference in {path}.")
                clips.append(clip)
            lengths = [len(clip["joint_pos"]) for clip in clips]
            self.names = [str(path.relative_to(cfg.motion_dir).with_suffix("")) for path in files]
            for key in fields:
                setattr(self, key, torch.cat([clip[key] for clip in clips]))
        else:
            source = MotionLib(cfg.motion_dir, self.fk.kinematics, device, cfg.source_fps)
            self.joint_pos, self.joint_vel = source.q, source.qd
            self.root_pos_w, self.root_quat_w = source.root_p, from_native(quat_from_matrix(source.root_R))
            self.root_lin_vel_w, self.root_ang_vel_w = source.root_v, source.root_w
            lengths, self.names = source.clip_len.tolist(), source.names
        self.motion_ends = torch.tensor(lengths, device=device).cumsum(0)
        self.motion_starts = self.motion_ends - torch.tensor(lengths, device=device)
        self.time_step_total = sum(lengths)

    def body_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute tracked body poses and velocities without retaining an all-body dataset."""
        return self.fk(
            self.root_pos_w[frames],
            self.root_quat_w[frames],
            self.joint_pos[frames],
            self.root_lin_vel_w[frames],
            self.root_ang_vel_w[frames],
            self.joint_vel[frames],
        )
