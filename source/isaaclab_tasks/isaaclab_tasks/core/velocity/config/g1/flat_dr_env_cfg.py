# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 flat-ground velocity tracking with a minimal, transfer-oriented randomization set.

``Isaac-Velocity-Flat-G1`` with a deployment-shaped randomization set and nothing else.

**Three randomized quantities, each spanning 0.5x to 2x nominal.** Contact friction, joint armature,
and actuator stiffness. These are the parameters a robot description is least able to pin down --
friction depends on the floor, armature is reflected rotor inertia through a gearbox, and the
effective stiffness of a motor's position loop drifts with load and temperature -- and they are the
ones a locomotion policy is most sensitive to. Everything else, including the observation noise, the
reset distribution, the pushes, the reward weights, the terminations, and the shipped implicit
actuators, is left exactly as the stock flat task has it.

**A five-frame observation window.** Randomization alone buys tolerance but leaves the policy unable
to tell which draw it received; the window is what lets it notice and adapt within an episode. The
two together are the usual implicit system-identification setup, which is why the window stays even
though the randomization set shrank.

**Two things a MuJoCo sim2sim proved were missing.** Joint friction is randomized to 0.3 N-m because
the MuJoCo G1 carries 0.2 on every joint and the trained range topped out at 0.05, and the leg and
foot commands are lagged 0-20 ms because the deployment path measured 12-16 ms of round trip and a
zero-lag policy loses two thirds of its survival time at 4 ms.

Ranges use the symmetric inverse form with a log-uniform draw, so 0.5x and 2x are equally likely and
the geometric mean sits at nominal. A plain uniform draw over the same interval would have an
arithmetic mean of 1.25x and quietly bias every episode toward the stiffer, grippier, heavier-rotor
end.
"""

import dataclasses

from isaaclab.actuators import ActuatorBaseCfg, DelayedPDActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg, ObservationsCfg

from .flat_env_cfg import G1FlatEnvCfg

_HISTORY_LENGTH = 5
"""Proprioceptive frames the policy sees, oldest first."""

_MIN_COMMAND_DELAY = 0
"""Minimum joint-command lag [physics steps]; 0 steps is 0 ms at ``sim.dt = 0.005``."""

_MAX_COMMAND_DELAY = 4
"""Maximum joint-command lag [physics steps]; 4 steps is 20 ms at ``sim.dt = 0.005``.

Latency is an actuator property, not a randomization luxury. A command reaches a G1 motor one bus
round trip behind the host, and even a MuJoCo sim2sim over DDS measured 12-16 ms of it. A policy
trained at zero lag has no tolerance for that: an open-loop sweep against this task's own checkpoint
dropped survival from 7.6 s at 0 ms to 3.5 s at 2 ms and 1.6 s at 4 ms. This span brackets what was
measured, so the policy has to learn to work across it.
"""

_TRUNK_MASS_SCALES = {
    # body -> (min, max) multiplier on the USD's mass, log-uniform.
    #
    # Not a randomization for its own sake: the USD and the official MJCF describe two different
    # robots, and the USD is the wrong one. Segment totals at the same default pose --
    #
    #   trunk (pelvis + waist links + torso + head)  USD 10.381 kg   MJCF 13.702 kg   +32%
    #   both legs                                    USD 15.028 kg   MJCF 14.372 kg    -4%
    #   both arms                                    USD  6.832 kg   MJCF  7.038 kg    +3%
    #   whole robot                                  USD 32.239 kg   MJCF 35.112 kg    +8.9%
    #
    # -- and a G1 EDU is specified at about 35 kg, so the MJCF is the one that matches hardware.
    # The legs and arms agree to a few percent; the entire discrepancy is the trunk.
    #
    # The stock ``add_base_mass`` already scales ``torso_link`` by 0.8x-1.25x, which sounds like it
    # covers this and does not: 6.340 kg x 1.25 is 7.93 kg against the real 9.60 kg, so the top of
    # the trained range sits 21% below hardware and the policy has never carried a real G1's torso.
    # The pelvis is not randomized at all and is 0.95 kg light, and the MJCF's waist yaw/roll links
    # (0.29 kg) have no USD counterpart at all.
    #
    # These spans are chosen to bracket both descriptions rather than to sit symmetrically around
    # the USD, because the USD is the outlier: 0.8x-1.6x of torso covers [5.07, 10.14] kg and
    # 0.8x-1.5x of pelvis covers [2.29, 4.29] kg, which absorbs the missing waist links too.
    "torso_link": (1 / 1.25, 1.6),
    "pelvis": (1 / 1.25, 1.5),
}
"""Trunk mass spans wide enough to contain the real robot, not just the USD's idea of it."""

_REL_STANDING_ENVS = 0.15
"""Fraction of environments commanded to hold still.

Standing is a separate mode from walking, not the limit of slow walking: the command goes exactly to
zero, and ``feet_air_time_positive_biped`` gates itself off below 0.1 m/s, so nothing pays for
stepping. The stock 0.02 gives that mode one episode in fifty.

The forward speed itself is not the problem -- ``lin_vel_x`` is uniform on [0, 1] and 43% of samples
land below 0.5 m/s. What is rare is the *all-zero* command: a logged 400 s play session put only 6.7%
of samples below 0.1 m/s total speed, essentially all of them the standing draws, because a small
``vx`` still comes with a ``vy`` uniform on [-0.5, 0.5].

The symptom matches: in MuJoCo this task's checkpoint walks 15 s at 0.25-1.0 m/s and falls at 1.9 s
on a zero command. Stiffening the waist hold and matching the trunk mass each fixed a real modelling
error and neither moved that number, which leaves the command distribution.
"""

_DELAYED_ACTUATOR_GROUPS = ("legs", "feet")
"""Actuator groups converted to delayed explicit PD. The arms and hands stay implicit.

Not a preference -- the arms group does not survive the conversion. G1 declares the shoulders, the
elbows and all fourteen finger joints in one group under a shared 300 N-m effort limit. Under the
solver's implicit PD that ceiling is never approached, but an explicit model is free to command it,
and 300 N-m into a finger whose armature is 1e-3 kg m^2 diverges inside a single physics step: a
zero-action rollout reaches 4e3 rad/s on the first env step. Locomotion is also where the lag
matters -- the arms are pinned near their default pose by the joint deviation penalties.
"""


_JOINT_POS_LIMITS = {
    # (lower, upper) [rad], the USD-and-MJCF intersection. Entries only where the two disagree by
    # more than 0.02 rad; the other nine joints already match.
    ".*_hip_pitch_joint": (-2.350, 2.880),
    "left_hip_roll_joint": (-0.260, 2.530),
    "right_hip_roll_joint": (-2.530, 0.260),
    ".*_knee_joint": (-0.087, 2.545),
    ".*_ankle_pitch_joint": (-0.680, 0.524),
    ".*_shoulder_pitch_joint": (-2.967, 2.670),
    ".*_elbow_pitch_joint": (-0.227, 2.094),
    ".*_elbow_roll_joint": (-1.972, 1.972),
}
"""Joint travel narrowed to what both the USD and the official MJCF allow.

Fourteen of the twenty-three driven joints disagree between the two descriptions, and not by a
rounding error: the elbow differs by 1.33 rad at the top, the knee by 0.25 at the bottom, the hip
roll by 0.44. Neither file is uniformly the more permissive one, so this is an intersection rather
than a copy.

The knee is why this matters. Its USD floor is -0.335 and its MJCF floor is -0.087, and the trained
policy parks the right knee against the floor 97% of the time -- the position target sits outside the
limit and the solver holds the joint there, which is a perfectly good way to lock a pose and costs
the policy nothing. Move that wall 0.25 rad and the same actions produce a different leg, which is
exactly what MuJoCo showed: the right knee pinned at -0.087 with 238 N·m of PD fighting the stop
while the left leg walked normally.

Narrowing the limits in training removes the option. A policy that cannot reach -0.335 cannot learn
to lean on it.
"""

_EFFORT_LIMITS = {
    "legs": {
        ".*_hip_yaw_joint": 88.0,
        ".*_hip_roll_joint": 88.0,
        ".*_hip_pitch_joint": 88.0,
        ".*_knee_joint": 139.0,
        "torso_joint": 88.0,
    },
    "feet": 50.0,
    "arms": {
        ".*_shoulder_.*": 25.0,
        ".*_elbow_.*": 25.0,
        # No hardware counterpart: g1_minimal.usd's three-finger hand is a separate Dex3 device.
        # Anything is better than the 300 N-m the group inherits; 5 N-m matches the wrist scale.
        ".*_(five|three|six|four|zero|one|two)_joint": 5.0,
    },
}
"""Per-joint torque ceilings [N·m], taken from the official G1 MJCF's ``ctrlrange``.

The shipped ``G1_CFG`` uses one blanket number per actuator group -- 300 for the legs, 300 for the
arms, 20 for the feet -- and none of the three matches the robot:

===================  ========  ==========  ============
joint                real G1   G1_CFG      error
===================  ========  ==========  ============
hip pitch/roll/yaw   88        300         3.4x too strong
knee                 139       300         2.2x too strong
ankle pitch/roll     50        20          2.5x too weak
shoulder, elbow      25        300         12x too strong
===================  ========  ==========  ============

Both directions hurt, and they hurt where it matters. Trained against a 300 N·m hip the policy
learns to call for torque the robot cannot deliver, and the excess is silently clipped on the way
out -- under-actuation with no error anywhere. Trained against a 20 N·m ankle it learns to balance
without the ankle authority it actually has, leaning on hip and stepping instead. That combination
matches what the open-loop replay showed: the load-bearing joints separated first, in load order.
"""


def _with_command_delay(actuator: ActuatorBaseCfg) -> DelayedPDActuatorCfg:
    """Re-type an actuator group as a delayed explicit PD group, leaving every gain unchanged.

    Copying the fields off the source config rather than restating them keeps this variant from
    drifting when the G1 asset gains are re-tuned upstream. Implicit actuators are integrated by the
    solver and cannot be lagged, so the group has to become explicit.

    Args:
        actuator: Source actuator group from the shipped robot config.

    Returns:
        The same group as a delayed explicit PD actuator.
    """
    fields = {f.name: getattr(actuator, f.name) for f in dataclasses.fields(actuator) if f.name != "class_type"}
    # An implicit actuator configured with only ``effort_limit_sim`` uses that solver clamp as its
    # model-facing limit too; an explicit one would fall back to the USD value instead.
    if fields.get("effort_limit") is None:
        fields["effort_limit"] = fields.get("effort_limit_sim")
    return DelayedPDActuatorCfg(**fields, min_delay=_MIN_COMMAND_DELAY, max_delay=_MAX_COMMAND_DELAY)


@configclass
class G1FlatDRObservationsCfg(ObservationsCfg):
    """The stock observation set, split so the actor only sees what a real G1 publishes.

    ``base_lin_vel`` is dropped from the policy group in :meth:`G1FlatDREnvCfg.__post_init__` and
    served to the critic instead. The G1 has no base linear velocity to give: the quantity would have
    to come from a state estimator that does not ship with the robot, and a policy that leans on it
    in simulation has nothing to read on hardware. The critic runs only in simulation and is never
    exported, so it can keep the ground truth and the value function stays well conditioned -- the
    standard asymmetric actor-critic split.

    The actor is not left guessing: inferring its own velocity from five frames of joint and IMU
    history is exactly the job the observation window was added for.
    """

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged state, simulation only."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class G1FlatDREventsCfg(EventsCfg):
    """The stock event set with three quantities randomized over 0.5x to 2x nominal.

    Every term is ``startup`` mode: friction is a property of the floor, armature is a property of
    the gearbox, and the motor's position-loop gain is a property of the unit. None of the three
    changes between two runs of the same robot on the same floor, so each environment is one draw and
    4096 environments are 4096 robots. The observation window still has work to do -- within an
    episode the policy has to infer which of those robots it is driving.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            # Stock values are 0.8 static / 0.6 dynamic, so these are those spans scaled by 0.5-2x.
            # The terrain contributes mu = 1.0 with a multiply combine mode, so they are also the
            # effective contact coefficients. Newton collapses friction to one number and ignores the
            # dynamic range and the bucket count; PhysX uses all three.
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            # The USD assumes frictionless joints. A harmonic drive is not, and the MuJoCo G1 model
            # carries 0.2 N-m of frictionloss on every joint -- four times the top of the range this
            # task used to randomize, so the sim2sim robot sat outside the trained distribution.
            # Absolute values, because there is no nonzero nominal to scale.
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.3),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    actuator_stiffness = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            # Stiffness only -- damping is left at nominal so the effective damping ratio moves with
            # the draw, which is what a real position loop does when its gain is off.
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class G1FlatDREnvCfg(G1FlatEnvCfg):
    """Stock G1 flat locomotion plus friction/armature/stiffness randomization and a 5-frame window."""

    events: G1FlatDREventsCfg = G1FlatDREventsCfg()
    observations: G1FlatDRObservationsCfg = G1FlatDRObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Widen the trunk mass draw to contain the real robot; see _TRUNK_MASS_SCALES. The parent
        # already points add_base_mass at torso_link, so this only replaces its range.
        self.events.add_base_mass.params["mass_distribution_params"] = _TRUNK_MASS_SCALES["torso_link"]
        self.events.add_pelvis_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "mass_distribution_params": _TRUNK_MASS_SCALES["pelvis"],
                "operation": "scale",
                "distribution": "log_uniform",
            },
        )

        # Give standing its own share of the episodes; see _REL_STANDING_ENVS.
        self.commands.base_velocity.rel_standing_envs = _REL_STANDING_ENVS

        # Not observable on hardware -- see G1FlatDRObservationsCfg. A term set to None is skipped by
        # the observation manager, the same way the flat parent drops the height scan.
        self.observations.policy.base_lin_vel = None

        # A group-level history overrides every term's, so the policy vector becomes five stacked
        # frames of the stock observation. History is buffered per term and the terms are
        # concatenated afterwards, so the layout is term by term -- ``[lin_vel(t-4..t),
        # ang_vel(t-4..t), ...]`` -- not frame by frame. A deployment stack has to reproduce that
        # order, not just the frame count.
        self.observations.policy.history_length = _HISTORY_LENGTH
        self.observations.policy.flatten_history_dim = True

        # Give the locomotion joints a per-episode command lag. The shipped implicit actuators are
        # integrated by the solver and cannot be delayed, so those groups become explicit PD; the
        # gains are copied across unchanged.
        missing = (set(_DELAYED_ACTUATOR_GROUPS) | set(_EFFORT_LIMITS)) - set(self.scene.robot.actuators)
        if missing:
            raise ValueError(
                f"Expected actuator groups {sorted(set(_DELAYED_ACTUATOR_GROUPS) | set(_EFFORT_LIMITS))} on"
                f" the G1 asset, but {sorted(missing)} are absent. The groups were renamed upstream;"
                " update the tables rather than letting the delay or the torque ceilings vanish."
            )
        # replace() rather than assignment: the actuator configs are shared with the module-level
        # G1_CFG, so mutating them in place would retune every other G1 task in the process.
        self.scene.robot.actuators = {
            name: (
                _with_command_delay(actuator.replace(effort_limit_sim=_EFFORT_LIMITS[name]))
                if name in _DELAYED_ACTUATOR_GROUPS
                else actuator.replace(effort_limit_sim=_EFFORT_LIMITS[name])
            )
            for name, actuator in self.scene.robot.actuators.items()
        }

        # Clamp joint travel to the USD-and-MJCF intersection. ArticulationCfg has no per-joint limit
        # field, so this goes through the sanctioned startup event: an "abs" operation with a
        # degenerate (x, x) range writes exactly x.
        for index, (expr, (lower, upper)) in enumerate(_JOINT_POS_LIMITS.items()):
            setattr(
                self.events,
                f"joint_limit_{index}",
                EventTerm(
                    func=mdp.randomize_joint_parameters,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", joint_names=expr),
                        "lower_limit_distribution_params": (lower, lower),
                        "upper_limit_distribution_params": (upper, upper),
                        "operation": "abs",
                        "distribution": "uniform",
                    },
                ),
            )
