# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 rough-terrain DR29 on the asset NVIDIA ships, with a pelvis-height floor.

:mod:`rough_dr29_env_cfg` trains against a locally converted URDF whose only collision geometry is
four spheres per foot. The robot cannot rest on anything, so nothing stops the policy from sinking:
over 10000 iterations the posture drifts from upright into a visible squat, while the success rate
keeps improving. Nothing in that task constrains base height at all.

The shipped ``Isaac/Robots/Unitree/G1/g1.usd`` carries colliders on the pelvis, both knees, the
waist and both wrists. On flat ground that turns the same drift into a much worse outcome: the
robot rests part of its weight on those colliders, the torso-contact termination never fires, and a
crouched non-walking gait survives the full episode -- ``success_rate`` 0.010. Adding a pelvis-height
floor recovers it to 1.000.

This config is that pair applied to rough DR29: the shipped asset, plus a floor at 0.4 m.

Two things the asset swap forces:

* **The height scanner moves.** :mod:`rough_dr29_env_cfg` points it several levels down, because a
  URDF conversion nests every body under its parent. The shipped asset is flat, so the stock
  ``Robot/torso_link`` is correct here.
* **The floor is measured against the terrain, not world z.**
  :func:`~isaaclab.envs.mdp.terminations.root_height_below_minimum` compares against world height and
  its docstring restricts it to flat ground; used on generated terrain it ends episodes for standing
  in a dip.

Select the asset with ``G1_29DOF_USD``; there is no separate asset config for it.
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCaster
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .rough_dr29_env_cfg import G1RoughDR29EnvCfg

_MINIMUM_PELVIS_HEIGHT = 0.4
"""Pelvis height above the terrain below which the episode ends [m]."""

_TARGET_PELVIS_HEIGHT = 0.72
"""Pelvis height a walking G1 holds [m]. WBC-AGILE's measured ``DEFAULT_PELVIS_HEIGHT``."""

_HEIGHT_REWARD_WEIGHT = -2.0
"""Weight on the height term.

Started at -10, which is what worked on flat ground, and it starved the main task on rough: at
iteration 3353 the terrain curriculum was at 4.08 against 5.5-5.7 for the other two variants,
episodes ran 765 steps against 914-947, and ``success_rate`` was still 0.000 where the others were
already at 0.76-0.80. Height itself was held fine -- the term sat at -0.040, a 6.3 cm deviation --
so the objective was being met at the cost of everything else. Holding 0.72 m is nearly free on
flat ground and expensive while climbing terrain; the weight has to reflect that.
"""


def pelvis_below_terrain_clearance(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the root sits less than ``minimum_height`` above the ground beneath it.

    The reference is the median ray hit rather than the mean: the scanner spans 1.6 x 1.0 m, so on
    broken ground a few rays land on a ledge or miss entirely, and a mean drags the reference far
    enough to end healthy episodes.

    Args:
        env: The environment.
        minimum_height: Clearance below which the episode ends [m].
        asset_cfg: Articulation whose root height is checked.
        sensor_cfg: Ray caster defining the ground beneath the robot.

    Returns:
        Boolean tensor, one entry per environment.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    return asset.data.root_pos_w.torch[:, 2] - ground < minimum_height


@configclass
class G1RoughDR29OfficialEnvCfg(G1RoughDR29EnvCfg):
    """DR29 rough on the shipped asset, with a pelvis-height floor."""

    def __post_init__(self):
        super().__post_init__()

        # The shipped asset keeps its links one level under the articulation root.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

        self.terminations.base_height = DoneTerm(
            func=pelvis_below_terrain_clearance,
            params={
                "minimum_height": _MINIMUM_PELVIS_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


@configclass
class G1RoughDR29OfficialRewardEnvCfg(G1RoughDR29OfficialEnvCfg):
    """The same asset, with the floor replaced by a reward that pulls toward 0.72 m.

    The floor version did not hold posture. Measured held-out over its 10000 iterations, pelvis
    height still fell 0.728 -> 0.590 after iteration 4000, and the share of episodes ending on the
    floor rose 8.3% -> 18.1%, dragging training reward from 7.79 down to 3.07 while
    ``Metrics/success_rate`` sat at 0.99 throughout. A floor only says "not below X"; the cheapest
    way to satisfy it is to sink until it binds and then pay the penalty.

    On flat ground the reward alone reached ``success_rate`` 0.992 with no floor at all, and without
    the locked-knee gait a floor-only variant produced. This is that configuration on rough terrain,
    with the target measured against the height scanner rather than world z.
    """

    def __post_init__(self):
        super().__post_init__()

        self.terminations.base_height = None
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=_HEIGHT_REWARD_WEIGHT,
            params={
                "target_height": _TARGET_PELVIS_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


@configclass
class G1RoughDR29OfficialTeacherEnvCfg(G1RoughDR29OfficialEnvCfg):
    """A *sighted* policy, to be distilled into a depth-camera student.

    Every other DR29 variant keeps ``height_scan`` in the critic-only group, because the actor is
    what gets deployed and a real G1 has no elevation map. A distillation teacher is the opposite
    case: only the actor is copied -- the critic exists to compute advantages during PPO and is
    thrown away -- so a teacher whose actor cannot see the terrain teaches the student nothing about
    using it, and the student learns to ignore its camera.

    So here the scan moves into the ``policy`` group. The agent config's
    ``obs_groups = {"actor": ["policy"], ...}`` then feeds it to the actor without further change.
    This policy is **not deployable** and is not meant to be; it is stage one of two.
    """

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
