# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for the Unitree G1 velocity-tracking environment on rough terrain.

Beside the standard environment this module holds the two stages of the depth pipeline: a teacher
whose actor reads a terrain height scan and the base linear velocity, and a student that replaces
both with a chest-mounted depth camera and is distilled from the teacher in the teacher's own
environment.
"""

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_assets import G1_MINIMAL_CFG

from ... import mdp
from ...velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
)


@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )


@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the Unitree G1 velocity-tracking environment on rough terrain."""

    rewards: G1Rewards = G1Rewards()

    def __post_init__(self):
        super().__post_init__()

        # physics
        self.sim.physics.newton_mjwarp.solver_cfg.njmax = 300
        # scene
        self.scene.robot = replace(G1_MINIMAL_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        # commands
        self.commands.base_velocity.vel_yaw_success_threshold = 0.8
        self.commands.base_velocity.marker_pos_offset = (0.0, 0.0, 0.75)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # events
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_com = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "torso_link"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


DEPTH_WIDTH = 64
"""Rendered depth width [px]. Small on purpose -- rendering is the cost of this stage."""

DEPTH_HEIGHT = 38
"""Rendered depth height [px], at the D435's 87:58.8 degree aspect."""

DEPTH_STACK = 3
"""Depth frames handed to the student as channels.

A forward-facing camera never sees the ground under or behind the robot, which the teacher's height
scan does. Stacking lets the student carry terrain it has already walked past.
"""

DEPTH_MAX_RANGE = 3.0
"""Range the depth is clipped to [m]. The D435's usable depth falls off well before its optical
limit, and this is also what a no-return maps to."""

DEPTH_OFFSET_POS = (0.0576, 0.0175, 0.4299)
"""Camera position on ``torso_link`` [m], measured on the robot."""

DEPTH_OFFSET_ROT = (0.9150, 0.0, -0.4035, 0.0)
"""Camera orientation as ``(w, x, y, z)`` in the ``world`` convention: 47.6 degrees below
horizontal, measured on the robot against 48.2."""


class DepthImageStack(ManagerTermBase):
    """The last :data:`DEPTH_STACK` depth frames as channels of one rank-4 observation.

    Isaac Lab's per-term ``history_length`` cannot be used here: it stacks on a new axis, yielding
    ``(B, T, C, H, W)``, and ``rsl_rl`` recognises only ``(B, C, H, W)`` as an image.
    """

    def __init__(self, cfg: ObsTerm, env) -> None:
        super().__init__(cfg, env)
        self._stack = CircularBuffer(
            max_len=DEPTH_STACK, batch_size=env.num_envs, device=env.device, stack_dim=1
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Without this a freshly reset environment keeps looking at the terrain its previous
        # episode ended on.
        self._stack.reset(env_ids)

    def __call__(self, env, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        """Stacked depth [m], shape ``(B, DEPTH_STACK, H, W)``."""
        frame = env.scene.sensors[sensor_cfg.name].data.output[data_type]
        frame = frame.torch if hasattr(frame, "torch") else frame
        # Rays that hit nothing come back as +inf. Clamping them to the far range says "nothing
        # within range", which is what the hardware reports; mapping them to zero would say
        # "something touching the lens".
        frame = torch.nan_to_num(frame, nan=DEPTH_MAX_RANGE, posinf=DEPTH_MAX_RANGE)
        # The rendered frame is rolled 180 degrees against the real camera: on flat ground its rows
        # run near-to-far where the hardware runs far-to-near, and an obstacle on the robot's left
        # lands on the right. Both axes, so it is a roll about the optical axis.
        self._stack.append(frame.permute(0, 3, 1, 2).flip((-2, -1)).contiguous())
        return self._stack.stacked.clone()


@configclass
class G1TeacherObservationsCfg(ObservationsCfg):
    """The privileged group the teacher trains on, beside the deployable group it is distilled into.

    ``policy`` holds exactly what a real G1 publishes. ``teacher`` adds the height scan and the base
    linear velocity, which it does not.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception, and nothing the hardware cannot measure."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    teacher: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()


@configclass
class G1RoughTeacherEnvCfg(G1RoughEnvCfg):
    """The teacher: PPO on the privileged observation."""

    observations: G1TeacherObservationsCfg = G1TeacherObservationsCfg()


@configclass
class G1DepthStudentObservationsCfg(G1TeacherObservationsCfg):
    """The teacher's groups plus the camera. ``policy`` and ``teacher`` are inherited unchanged."""

    @configclass
    class DepthCfg(ObsGroup):
        """Rank 4, and kept on its own so ``rsl_rl`` routes it through the convolutional encoder."""

        depth = ObsTerm(
            func=DepthImageStack,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
            clip=(0.0, DEPTH_MAX_RANGE),
            scale=1.0 / DEPTH_MAX_RANGE,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()


@configclass
class G1RoughDepthStudentEnvCfg(G1RoughTeacherEnvCfg):
    """The student: the teacher's environment, plus the camera it is given instead of the scan."""

    observations: G1DepthStudentObservationsCfg = G1DepthStudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/depth_camera",
            update_period=self.decimation * self.sim.dt,
            width=DEPTH_WIDTH,
            height=DEPTH_HEIGHT,
            data_types=["distance_to_image_plane"],
            # 87 x 58.8 degrees, the D435's depth field of view.
            spawn=PinholeCameraCfg(focal_length=1.93, horizontal_aperture=3.58, clipping_range=(0.1, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=DEPTH_OFFSET_POS, rot=DEPTH_OFFSET_ROT, convention="world"),
        )
