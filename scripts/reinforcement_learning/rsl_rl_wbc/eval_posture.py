# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measure the posture and gait symmetry of a WBC-AGILE G1 velocity policy.

``success_rate`` and the time-out share cannot see a splayed or crouching gait -- a policy that
walks 30 s without falling scores the same whether its pelvis is level or rolled 7 degrees. This
measures the things that do move, under the one condition where every part of this robot should be
mirror-symmetric: **flat ground with the command pinned to a straight walk**.

Confounders this removes, each of which has produced a wrong answer before:

* generated terrain legitimately breaks symmetry -- so ``terrain_type="plane"``;
* the task's own command samples lateral and yaw components plus 20% standing environments -- so
  the command is pinned and ``rel_standing_envs`` is zeroed;
* pushes and the interval force events break it too -- so they are removed;
* the harness is faded out by a *curriculum*, so with curricula disabled it would come back at full
  strength and hold the robot up -- so the harness action is deleted, as WBC's own ``eval()`` does;
* the world-frame mirror test reports a symmetric asset as 0.4 m asymmetric because the root
  carries a random yaw at reset -- so everything is measured in the base frame;
* averaging a statistic over all joints deflates it ~20x because of the hands -- this robot has no
  hands (WBC strips them) but the arms are still driven by a random-pose action, so only the twelve
  leg joints are used.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Velocity-G1-WBC-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--lin_vel_x", type=float, default=0.5, help="Pinned forward command [m/s].")
parser.add_argument("--warmup", type=int, default=200, help="Control steps discarded before measuring.")
parser.add_argument("--steps", type=int, default=1000, help="Control steps measured.")

import cli_args  # isort: skip  # noqa: E402  -- must not import isaaclab_tasks before the app starts

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# ``--checkpoint`` comes from cli_args (AGILE's shared RSL-RL arguments), where it is optional.
if not args_cli.checkpoint:
    parser.error("--checkpoint is required: pass a model_*.pt to measure")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks.contrib.wbc_velocity.isaaclab_extras  # noqa: F401, E402
import isaaclab_tasks.contrib.wbc_velocity  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from isaaclab_tasks.contrib.wbc_velocity.rsl_rl import (  # isort: skip  # noqa: E402
    RslRlVecEnvWrapper,
    make_rsl_rl_runner,
)

# Left/right leg joint pairs and the sign the right joint takes under a mirror. Roll and yaw flip;
# pitch and the knee do not. A pair that is truly symmetric satisfies q_L == sign * q_R.
MIRROR_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
]
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


def pin_to_a_straight_walk_on_flat_ground(env_cfg):
    """Strip everything that would legitimately break symmetry, so what is left is the policy."""
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.curriculum = None

    cmd = env_cfg.commands.base_velocity
    cmd.rel_standing_envs = 0.0
    cmd.heading_command = False
    cmd.resampling_time_range = (1.0e9, 1.0e9)  # never resample: one command for the whole episode
    cmd.ranges.lin_vel_x = (args_cli.lin_vel_x, args_cli.lin_vel_x)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)

    # The harness is removed by curriculum during training; with curricula off it would return at
    # full strength and hold the pelvis up, which is exactly what we are trying to measure.
    if hasattr(env_cfg.actions, "harness"):
        del env_cfg.actions.harness

    for event in ("push_robot", "apply_external_force_torque", "apply_external_force_torque_extremities"):
        if getattr(env_cfg.events, event, None) is not None:
            setattr(env_cfg.events, event, None)

    # Observation noise is a training device; it only adds variance to a measurement.
    env_cfg.observations.policy.enable_corruption = False
    return env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg = pin_to_a_straight_walk_on_flat_ground(env_cfg)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = make_rsl_rl_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] loaded {args_cli.checkpoint}")

    robot = env.unwrapped.scene["robot"]
    contacts = env.unwrapped.scene["contact_forces"]
    foot_ids = [robot.body_names.index(name) for name in FEET]
    contact_ids = [contacts.body_names.index(name) for name in FEET]
    pair_ids = [(robot.joint_names.index(l), robot.joint_names.index(r), s) for l, r, s in MIRROR_PAIRS]

    # Everything is accumulated *signed* and reduced at the end. A walking gait oscillates about its
    # posture, so taking |x| per step and averaging measures the swing amplitude, not the bias --
    # and the two legs are half a cycle out of phase, so an instantaneous |q_L - mirror(q_R)| reads
    # ~13 deg on a perfectly symmetric gait. The persistent lean is what we are after.
    acc = {k: 0.0 for k in ("roll", "pitch", "height", "vel_x", "foot_y_sum")}
    roll_abs = 0.0
    airborne = torch.zeros(2, device=env.unwrapped.device)
    pair_l = torch.zeros(len(pair_ids), device=env.unwrapped.device)
    pair_r = torch.zeros(len(pair_ids), device=env.unwrapped.device)
    n = 0

    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(args_cli.warmup + args_cli.steps):
            obs, _, _, _ = env.step(policy(obs))
            if step < args_cli.warmup:
                continue
            n += 1

            quat = robot.data.root_quat_w.torch
            roll, pitch, _ = euler_xyz_from_quat(quat)
            # euler_xyz_from_quat returns [0, 2pi); fold to (-pi, pi] before averaging magnitudes,
            # otherwise a -1 degree roll averages in as +359.
            roll = torch.atan2(torch.sin(roll), torch.cos(roll))
            pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
            acc["roll"] += roll.mean().item()
            roll_abs += roll.abs().mean().item()
            acc["pitch"] += pitch.mean().item()
            acc["height"] += robot.data.root_pos_w.torch[:, 2].mean().item()
            acc["vel_x"] += robot.data.root_lin_vel_b.torch[:, 0].mean().item()

            # Feet in the base frame: the root yaw at reset is random, so world-frame y is noise.
            feet_w = robot.data.body_pos_w.torch[:, foot_ids, :]
            feet_b = quat_apply_inverse(
                quat.unsqueeze(1).expand(-1, 2, -1), feet_w - robot.data.root_pos_w.torch.unsqueeze(1)
            )
            # A symmetric stance has the two feet equally far either side of the base x-axis, so
            # their y-coordinates sum to zero.
            acc["foot_y_sum"] += (feet_b[:, 0, 1] + feet_b[:, 1, 1]).mean().item()

            forces = contacts.data.net_forces_w.torch[:, contact_ids, :].norm(dim=-1)
            airborne += (forces < 1.0).float().mean(dim=0)

            q = robot.data.joint_pos.torch
            for k, (li, ri, sign) in enumerate(pair_ids):
                pair_l[k] += q[:, li].mean()
                pair_r[k] += sign * q[:, ri].mean()

    deg = 180.0 / math.pi
    print("\n" + "=" * 74)
    print(f"  {args_cli.task}  |  {args_cli.checkpoint.split('/')[-1]}")
    print(f"  flat ground, command pinned to lin_vel_x={args_cli.lin_vel_x} m/s, "
          f"{args_cli.num_envs} envs x {args_cli.steps} steps")
    print("=" * 74)
    print(f"  pelvis roll (lean)     {acc['roll'] / n * deg:7.2f} deg    (0 = level)")
    print(f"  pelvis roll (swing)    {roll_abs / n * deg:7.2f} deg    mean|roll|, gait amplitude")
    print(f"  pelvis pitch (lean)    {acc['pitch'] / n * deg:7.2f} deg")
    print(f"  pelvis height          {acc['height'] / n:7.3f} m      (commanded 0.720)")
    print(f"  forward velocity       {acc['vel_x'] / n:7.3f} m/s    (commanded {args_cli.lin_vel_x:.3f})")
    print(f"  foot lateral imbalance {acc['foot_y_sum'] / n:7.4f} m      (0 = stance centred)")
    air = (airborne / n).tolist()
    ratio = air[0] / air[1] if air[1] > 1e-9 else float("inf")
    print(f"  airborne share L/R     {air[0]:7.3f} / {air[1]:.3f}   ratio {ratio:.3f}  (1.0 = even)")
    asym = ((pair_l - pair_r) / n * deg).tolist()
    print("  per-joint-pair asymmetry [deg]  (time-averaged left minus mirrored right):")
    for k, (l, _, _) in enumerate(MIRROR_PAIRS):
        print(f"      {l.replace('left_', '').replace('_joint', ''):<14} {asym[k]:+7.2f}")
    worst = max(range(len(asym)), key=lambda k: abs(asym[k]))
    print(f"  worst pair: {MIRROR_PAIRS[worst][0].replace('left_', '').replace('_joint', '')} "
          f"at {asym[worst]:+.2f} deg")
    print("=" * 74 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
