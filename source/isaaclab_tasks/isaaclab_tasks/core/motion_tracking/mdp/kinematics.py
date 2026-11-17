# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable articulated-body poses and velocities from the task's USD joint frames."""

import torch

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.assets import Articulation
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis


class ArticulationKinematics:
    """Forward kinematics for a floating articulation with revolute joints.

    Joint frames come from the same USD and variants as the simulated robot. The root pose,
    joint angles and velocities remain differentiable; the asset is read only at construction.
    """

    def __init__(self, robot: Articulation, body_names: list[str], device: str):
        stage = Usd.Stage.Open(retrieve_file_path(robot.cfg.spawn.usd_path))
        stage.SetEditTarget(stage.GetSessionLayer())
        root = stage.GetDefaultPrim()
        for name, selection in (robot.cfg.spawn.variants or {}).items():
            root.GetVariantSet(name).SetVariantSelection(selection)
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or robot.cfg.spawn.scale not in (None, (1.0, 1.0, 1.0)):
            raise ValueError("Motion kinematics requires an unscaled asset authored in meters.")
        names = robot.joint_names
        pending = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint) or not UsdPhysics.Joint(prim).GetJointEnabledAttr().Get():
                continue
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parents or not children:
                continue
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError(f"Unsupported kinematic joint: {prim.GetPath()}.")
            if prim.GetName() not in names:
                raise ValueError(f"USD joint {prim.GetName()} is absent from the simulated articulation.")
            pending.append((prim.GetName(), parents[0].name, children[0].name, joint))
        self.body_names = [robot.body_names[0]]
        self.parents, self.joints = [], []
        positions0, positions1, rotations0, rotations1, axes = [], [], [], [], []
        while pending:
            remaining = []
            for name, parent, child, joint in pending:
                if parent not in self.body_names:
                    remaining.append((name, parent, child, joint))
                    continue
                self.parents.append(self.body_names.index(parent))
                self.joints.append(names.index(name))
                self.body_names.append(child)
                positions0.append(tuple(joint.GetLocalPos0Attr().Get()))
                positions1.append(tuple(joint.GetLocalPos1Attr().Get()))
                for destination, value in (
                    (rotations0, joint.GetLocalRot0Attr().Get()),
                    (rotations1, joint.GetLocalRot1Attr().Get()),
                ):
                    destination.append((*value.GetImaginary(), value.GetReal()))
                axis = UsdPhysics.RevoluteJoint(joint.GetPrim()).GetAxisAttr().Get()
                axes.append(tuple(float(axis == letter) for letter in "XYZ"))
            if len(remaining) == len(pending):
                raise ValueError("USD joint graph does not form a connected floating articulation.")
            pending = remaining
        if sorted(self.joints) != list(range(len(names))):
            raise ValueError("USD kinematics must cover each simulated joint exactly once.")
        self.tracked = [self.body_names.index(name) for name in body_names]
        self.positions0 = torch.tensor(positions0, device=device)
        self.positions1 = torch.tensor(positions1, device=device)
        self.rotations0 = matrix_from_quat(torch.tensor(rotations0, device=device))
        self.rotations1 = matrix_from_quat(torch.tensor(rotations1, device=device))
        self.axes = torch.tensor(axes, device=device)

    def forward(self, position, quaternion, linear_velocity, angular_velocity, joint_pos, joint_vel):
        """Return tracked world positions, rotation matrices, linear and angular velocities."""
        positions, rotations = [position], [matrix_from_quat(quaternion)]
        velocities, angular = [linear_velocity], [angular_velocity]
        for i, (parent, joint) in enumerate(zip(self.parents, self.joints)):
            pp, rp, vp, wp = positions[parent], rotations[parent], velocities[parent], angular[parent]
            offset = (rp @ self.positions0[i, :, None]).squeeze(-1)
            pivot = pp + offset
            frame = rp @ self.rotations0[i]
            rotation = frame @ matrix_from_quat(quat_from_angle_axis(joint_pos[..., joint], self.axes[i]))
            rotation = rotation @ self.rotations1[i].T
            axis = (frame @ self.axes[i, :, None]).squeeze(-1)
            w = wp + axis * joint_vel[..., joint, None]
            child_offset = (rotation @ self.positions1[i, :, None]).squeeze(-1)
            positions.append(pivot - child_offset)
            rotations.append(rotation)
            angular.append(w)
            velocities.append(vp + torch.cross(wp, offset, dim=-1) - torch.cross(w, child_offset, dim=-1))
        return (
            torch.stack([positions[i] for i in self.tracked], -2),
            torch.stack([rotations[i] for i in self.tracked], -3),
            torch.stack([velocities[i] for i in self.tracked], -2),
            torch.stack([angular[i] for i in self.tracked], -2),
        )
