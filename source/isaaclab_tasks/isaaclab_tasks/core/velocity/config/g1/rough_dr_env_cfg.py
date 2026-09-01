# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 rough-terrain velocity tracking, randomized for transfer to hardware.

``Isaac-Velocity-Rough-G1`` trains one exact robot on one exact floor with perfect state. A policy
learned that way reads its own base linear velocity and a terrain height map -- neither of which a
real G1 publishes -- and it has never met the actuator lag, the calibration offsets, or the mass and
friction spread that separate a physical unit from its USD file. This variant closes those gaps
along four axes.

**What the policy is allowed to see.** The ``policy`` group keeps only what the robot can stream at
50 Hz: IMU angular velocity and gravity direction, joint encoders, the velocity command, and its own
last action. ``base_lin_vel`` and ``height_scan`` move to a ``critic`` group that exists only in
simulation. That is the standard asymmetric actor-critic split: the value function stays
well-conditioned while the actor stays deployable.

**What the policy has to infer.** Every observation carries five frames of history, and the dynamics
move underneath it -- per episode for actuator gains and command lag, per environment for mass,
friction, armature, and joint friction. Neither ingredient works alone. Randomization without a
window leaves the policy unable to tell which draw it received; a window without randomization
carries no information worth using. Together they are the usual implicit system-identification
setup.

**What the sensors get wrong.** Each observation carries per-step noise *and* a bias re-drawn on
reset. The bias is what a fixed IMU mounting error or an uncalibrated encoder zero actually looks
like: constant within a run, different on the next robot. Zero-mean per-step noise alone teaches the
policy to average the error away, which is the wrong lesson -- on hardware the error does not
average away.

**When the command arrives.** The leg and foot joints run an explicit PD loop lagged by a per-episode
0-20 ms, the way a command actually reaches a motor driver over a bus. The arms and hands stay on the
solver's implicit PD; see :data:`_DELAYED_ACTUATOR_GROUPS` for why that split is forced rather than
chosen.

Expect this to train slower than the baseline: the actor is blind and the ground keeps shifting. See
the iteration count on :class:`~isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg.G1RoughDRPPORunnerCfg`.
"""

import dataclasses

from isaaclab.actuators import ActuatorBaseCfg, DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg, TerminationsCfg
from isaaclab_tasks.utils import preset

from .rough_env_cfg import G1RoughEnvCfg

##
# Actuation
##

# The real G1 closes its joint PD loop on the motor driver, one bus round-trip behind the host that
# evaluates the policy. ``sim.dt`` is 5 ms, so 0-4 physics steps spans 0-20 ms and brackets the
# host-to-motor lag of a typical DDS + CAN stack. ``DelayedPDActuator`` re-draws the lag per
# environment on every reset, which turns latency into one more per-episode parameter for the
# observation window to identify instead of a fixed offset the policy can silently bake in.
_MIN_COMMAND_DELAY = 0
"""Minimum joint-command lag [physics steps]; 0 steps is 0 ms at ``sim.dt = 0.005``."""

_MAX_COMMAND_DELAY = 4
"""Maximum joint-command lag [physics steps]; 4 steps is 20 ms at ``sim.dt = 0.005``."""

_DELAYED_ACTUATOR_GROUPS = ("legs", "feet")
"""Actuator groups converted to delayed explicit PD. The arms and hands stay implicit.

Not a preference -- the arms group does not survive the conversion. G1 declares the shoulders, the
elbows and all fourteen finger joints in one group under a shared 300 N-m effort limit. Under the
solver's implicit PD that ceiling is never approached, but an explicit model is free to command it,
and 300 N-m into a finger whose armature is 1e-3 kg m^2 diverges inside a single physics step: a
zero-action rollout reaches 4e3 rad/s on the first env step and goes non-finite on the second.
Restricting the conversion to the legs and feet holds the same rollout at 12 rad/s, against 8 rad/s
for the fully implicit baseline.

Locomotion is also where the lag matters. The arms are pinned near their default pose by the joint
deviation penalties and are not part of the balance loop the policy has to close through the delay.
"""


def _with_command_delay(actuator: ActuatorBaseCfg) -> DelayedPDActuatorCfg:
    """Re-type an actuator group as a delayed explicit PD group, leaving every gain untouched.

    Copying the fields off the source config rather than restating them keeps this variant from
    drifting silently when the G1 asset gains are re-tuned upstream.

    Two things change besides the delay. Implicit actuators are integrated by the solver and cannot
    be lagged, so the group has to become explicit. And an explicit model keeps its PD gains in
    tensors rather than in solver state, which is what makes re-randomizing them every episode cheap
    enough to actually do.

    Args:
        actuator: Source actuator group, typically an implicit one from the shipped robot config.

    Returns:
        The same group as a delayed explicit PD actuator.
    """
    fields = {f.name: getattr(actuator, f.name) for f in dataclasses.fields(actuator) if f.name != "class_type"}
    # An implicit actuator configured with only ``effort_limit_sim`` uses that solver clamp as its
    # model-facing limit as well. Explicit actuators do not: they would fall back to the USD value
    # and quietly saturate somewhere else.
    if fields.get("effort_limit") is None:
        fields["effort_limit"] = fields.get("effort_limit_sim")
    return DelayedPDActuatorCfg(**fields, min_delay=_MIN_COMMAND_DELAY, max_delay=_MAX_COMMAND_DELAY)


##
# Sensor error
##

# Each model is per-step noise plus a bias re-drawn on reset, with the bias sampled per component so
# every axis and every joint gets its own offset.

_GYRO_NOISE = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Unoise(n_min=-0.2, n_max=0.2),
    bias_noise_cfg=Gnoise(mean=0.0, std=0.05, operation="abs"),
)
"""Base angular velocity error [rad/s]: white noise plus a turn-on gyro bias."""

_GRAVITY_NOISE = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
    bias_noise_cfg=Gnoise(mean=0.0, std=0.03, operation="abs"),
)
"""Gravity-direction error [-]: 0.03 on a unit vector is roughly 1.7 deg of fixed IMU mounting tilt."""

_JOINT_POS_NOISE = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Unoise(n_min=-0.01, n_max=0.01),
    bias_noise_cfg=Gnoise(mean=0.0, std=0.015, operation="abs"),
)
"""Joint position error [rad]: encoder noise plus the per-joint zero offset left by calibration."""

_HISTORY_LENGTH = 5
"""Proprioceptive frames the actor sees, oldest first."""


##
# MDP settings
##


@configclass
class G1DRObservationsCfg:
    """Deployable proprioception for the actor, privileged state for the critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Everything a real G1 can stream at 50 Hz, and nothing else.

        History is buffered per term and the terms are concatenated afterwards, so the vector is laid
        out term by term -- ``[ang_vel(t-4..t), gravity(t-4..t), ...]`` -- not frame by frame. A
        deployment stack has to reproduce that order, not just the frame count.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=_GYRO_NOISE)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=_GRAVITY_NOISE)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=_JOINT_POS_NOISE)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = _HISTORY_LENGTH
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """State the simulator knows and the robot does not.

        Base linear velocity needs a state estimator the G1 does not ship, and the height scan needs
        an elevation map built from a depth sensor. Both are free during training and neither is ever
        exported with the policy.

        A blind actor on rough terrain is the harder problem of the two. If the deployment stack does
        run elevation mapping, move ``height_scan`` up to :class:`PolicyCfg` and give it a noise term.
        If it does not, the alternative to training blind from scratch is to distil a height-scan
        teacher into a blind student, the way ``Isaac-Velocity-Flat-AnymalD`` wires up
        ``rsl_rl_distillation_cfg_entry_point``.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1DREventsCfg(EventsCfg):
    """Dynamics randomization sized for a physical G1.

    Terms are split by what the quantity physically is. Link masses, the floor, reflected rotor
    inertia and joint friction do not change between two runs of the same robot, so they are drawn
    once per environment at startup -- 4096 environments is 4096 robots. Actuator gains and external
    disturbances do change run to run, so they are re-drawn on reset; those are the parameters the
    observation window has to track within an episode.

    Ranges use the symmetric inverse form ``(1/k, k)`` with a log-uniform draw, so the geometric mean
    stays at nominal and a heavier-than-nominal draw is exactly as likely as its lighter mirror.
    """

    # -- startup: unit-to-unit and floor-to-floor spread

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # The terrain contributes mu = 1.0 with a multiply combine mode, so these are the
            # effective contact coefficients: a polished lab floor at the low end, rubber matting at
            # the high end. Newton collapses friction to one coefficient and ignores the dynamic
            # range and the bucket count; PhysX uses all three.
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (1 / 1.15, 1.15),
            "operation": "scale",
            "distribution": "log_uniform",
            "recompute_inertia": True,
        },
    )

    torso_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Absolute kilograms rather than a ratio: this is the battery, the compute box and the
            # sensor mast a deployed G1 carries and a benchmark G1 does not.
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    torso_com = preset(
        default=EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.05, 0.05)},
            },
        ),
        # MJWarp bakes the center of mass into the model at build time, so there is no runtime
        # setter to randomize. The payload term above still shifts the torso inertia.
        newton_mjwarp=None,
    )

    joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # Armature is reflected rotor inertia through the gearbox. It is the least well
            # identified number in any robot description and MJWarp is particularly sensitive to it,
            # since it enters the generalized mass matrix directly.
            "armature_distribution_params": (1 / 1.4, 1.4),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            # The shipped asset assumes frictionless joints. A harmonic drive is not frictionless,
            # and how much stiction a given unit has depends on assembly and on how many hours it has
            # run. Set as an absolute value because there is no nonzero nominal to scale.
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    # -- reset: run-to-run spread

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # Stiffness stands in for motor strength as well: for a position-controlled joint the two
            # are the same knob. Damping gets the wider range because it is identified worst and
            # because too little of it is what makes a policy chatter on hardware. Re-drawing this
            # every reset is affordable on both backends -- the explicit legs and feet keep their
            # gains in tensors, and the implicit arms write through a device-side binding.
            "stiffness_distribution_params": (1 / 1.35, 1.35),
            "damping_distribution_params": (1 / 1.6, 1.6),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            # Held for the whole episode rather than applied as an impulse: this is a tether, a
            # dragging cable, or a payload mounted off-center, not a shove. 20 N is roughly 6 percent
            # of the robot's weight.
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-20.0, 20.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    # -- interval: disturbances

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        # Roughly twice as often and half again as hard as the baseline. A biped that has only been
        # pushed once per episode has not learned to recover, it has learned to wait.
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "yaw": (-0.5, 0.5)}},
    )


@configclass
class G1DRTerminationsCfg(TerminationsCfg):
    """Termination terms for the MDP."""

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    """End the episode once the torso passes 1.0 rad from upright.

    The inherited contact term only fires after the torso has already hit the ground. Cutting the
    episode at the point of no return keeps the policy from learning acrobatic saves that a real G1
    would not survive attempting.
    """


##
# Environment configuration
##


def apply_g1_dr_overrides(cfg: G1RoughEnvCfg) -> None:
    """Apply the two randomization overrides that cannot be expressed declaratively, in place.

    Everything else this variant changes is a config field the subclass can simply redeclare. These
    two cannot be, because :meth:`G1RoughEnvCfg.__post_init__` has already run by the time the
    subclass gets control and would undo them:

    * the shipped implicit actuator groups have to be re-typed as delayed explicit PD, and the robot
      config only exists after the parent assigns it;
    * the parent's torso-only ``add_base_mass`` term has to be retired in favour of the per-link
      ``link_mass`` / ``torso_payload`` / ``torso_com`` terms on :class:`G1DREventsCfg`, and it cannot
      simply be set to ``None`` on the class because the parent dereferences it.

    Shared by :class:`G1RoughDREnvCfg` and the flat variant so the two cannot drift apart.

    Args:
        cfg: Environment config to modify. Must already have run the G1 parent's post-init.

    Raises:
        ValueError: If the G1 asset no longer declares the actuator groups named in
            :data:`_DELAYED_ACTUATOR_GROUPS`.
    """
    missing = set(_DELAYED_ACTUATOR_GROUPS) - set(cfg.scene.robot.actuators)
    if missing:
        raise ValueError(
            f"Expected actuator groups {sorted(_DELAYED_ACTUATOR_GROUPS)} on the G1 asset, but"
            f" {sorted(missing)} are absent. The groups were renamed upstream; update"
            " _DELAYED_ACTUATOR_GROUPS rather than letting the command delay silently disappear."
        )
    cfg.scene.robot.actuators = {
        name: _with_command_delay(actuator) if name in _DELAYED_ACTUATOR_GROUPS else actuator
        for name, actuator in cfg.scene.robot.actuators.items()
    }
    cfg.events.add_base_mass = None


@configclass
class G1RoughDREnvCfg(G1RoughEnvCfg):
    """Randomized, partially observable G1 rough-terrain locomotion."""

    observations: G1DRObservationsCfg = G1DRObservationsCfg()
    events: G1DREventsCfg = G1DREventsCfg()
    terminations: G1DRTerminationsCfg = G1DRTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        apply_g1_dr_overrides(self)
