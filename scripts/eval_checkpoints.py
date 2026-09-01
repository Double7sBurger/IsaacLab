# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Score every checkpoint of a run on a fixed held-out suite.

Training curves cannot answer "when did this start overfitting". ``Metrics/success_rate`` in the
training log is measured on the environments the policy is currently training in: the terrain
curriculum has already promoted the ones that are doing well, and each environment's domain
randomization draw is the one it was trained on. A checkpoint can look better simply because the
curriculum moved.

This evaluates every checkpoint against conditions none of them trained on, held fixed across
checkpoints so the numbers are comparable:

* **A different randomization draw.** The DR terms are startup mode, so one seed is one population
  of robots. ``--seed`` differs from training's, giving robots with friction, armature, joint
  friction and actuator gains the policy has never seen.
* **No curriculum.** Terrain level is pinned, so every checkpoint is scored on the same ground
  rather than on whatever level it had earned.
* **The same command schedule.** Velocity commands are resampled from a generator seeded per
  episode index, not per checkpoint, so checkpoint N and checkpoint M are asked to do the same
  things in the same order.

Reported per checkpoint: success rate, mean tracking error, mean episode length, and the fraction
of episodes ending in a fall. A run that is overfitting shows training success flat or rising while
held-out success falls.

Example::

    uv run python scripts/eval_checkpoints.py \\
        --task Isaac-Velocity-Rough-G1-DR29 \\
        --run_dir logs/rsl_rl/g1_rough_dr29/2026-08-31_14-56-34 \\
        --every 500 --episodes 64 --terrain_level 4
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import re
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.common import add_frontend_args

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="Task the checkpoints were trained on.")
parser.add_argument("--run_dir", type=str, required=True, help="Directory holding ``model_*.pt``.")
parser.add_argument("--every", type=int, default=500, help="Evaluate one checkpoint every N iterations.")
parser.add_argument("--episodes", type=int, default=64, help="Episodes per checkpoint (rounded up to num_envs).")
parser.add_argument("--num_envs", type=int, default=64, help="Environments run in parallel.")
parser.add_argument("--seed", type=int, default=12345, help="Held-out seed; must differ from training's.")
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="Pin every environment to this terrain level. Default: leave the task's own setting.",
)
parser.add_argument("--out", type=str, default=None, help="CSV output path. Default: <run_dir>/eval_heldout.csv")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
add_frontend_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab_rl.entrypoints.common import create_isaaclab_env  # noqa: E402, F811
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _checkpoints(run_dir: str, every: int) -> list[tuple[int, str]]:
    """Return ``(iteration, path)`` for the checkpoints to score, oldest first."""
    found = []
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"model_(\d+)\.pt", name)
        if m:
            found.append((int(m.group(1)), os.path.join(run_dir, name)))
    found.sort()
    if not found:
        raise FileNotFoundError(f"No model_*.pt under {run_dir}")
    # Always keep the last one: it is the checkpoint anyone would otherwise ship.
    keep = [c for c in found if c[0] % every == 0]
    if found[-1] not in keep:
        keep.append(found[-1])
    return keep


@hydra_task_config(args_cli.task, args_cli.agent, play_mode=True)
def main(env_cfg, agent_cfg):
    """Score each checkpoint and write one CSV row per checkpoint."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Hold the ground fixed. Without this each checkpoint would be scored on whatever terrain level
    # the curriculum had reached, which is itself a function of how well training was going.
    if args_cli.terrain_level is not None and getattr(env_cfg.scene, "terrain", None) is not None:
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.curriculum = False
            env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level
    if getattr(env_cfg, "curriculum", None) is not None:
        for term in [t for t in vars(env_cfg.curriculum) if not t.startswith("_")]:
            setattr(env_cfg.curriculum, term, None)

    checkpoints = _checkpoints(args_cli.run_dir, args_cli.every)
    out_path = args_cli.out or os.path.join(args_cli.run_dir, "eval_heldout.csv")
    print(f"[eval] {len(checkpoints)} checkpoints, seed {args_cli.seed}, {args_cli.episodes} episodes each")

    with launch_simulation(env_cfg, args_cli):
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata as metadata  # noqa: PLC0415

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

        rows = []
        for iteration, path in checkpoints:
            runner.load(path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            # Reset before scoring: otherwise the first episodes of this checkpoint continue from
            # wherever the previous checkpoint's rollout left the robots, and get attributed here.
            env.unwrapped.reset()
            rows.append(_score(env, policy, iteration, args_cli.episodes))
            print(
                f"[eval] iter {iteration:>6}  success {rows[-1]['success_rate']:.3f}"
                f"  err_vel {rows[-1]['error_vel_xy']:.3f}"
                f"  ep_len {rows[-1]['ep_len']:.1f}"
                f"  pelvis_h {rows[-1]['pelvis_height']:.3f}"
                f"  fell {rows[-1]['fall_rate']:.3f}"
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] wrote {out_path}")


def _t(x):
    """Isaac Lab hands out plain tensors in some places and ProxyArrays in others."""
    return x.torch if hasattr(x, "torch") else x


def _score(env, policy, iteration: int, episodes: int) -> dict:
    """Run ``episodes`` episodes and summarize. Episodes are counted as they terminate."""
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    max_len = int(env.unwrapped.max_episode_length)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    # The task's own success definition: the episode-mean tracking error is under both thresholds.
    # ``command_manager`` only fills ``metrics`` during reset and the manager zeroes it immediately
    # afterwards, so reading that dict per step yields zeros -- recompute it here instead, with the
    # same formula and the same thresholds the term uses.
    cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
    thr_xy = cmd_term.cfg.vel_xy_success_threshold
    thr_yaw = cmd_term.cfg.vel_yaw_success_threshold

    done_count = 0
    lengths, fells, errs, heights, tracked = [], [], [], [], []
    steps = torch.zeros(num_envs, device=device)
    err_sum = torch.zeros(num_envs, device=device)
    yaw_err_sum = torch.zeros(num_envs, device=device)
    # Pelvis height above its own terrain, averaged over the episode. Posture drifts over training
    # -- a policy can keep its success rate while sinking into a crouch -- and nothing in the
    # training log records it.
    height_sum = torch.zeros(num_envs, device=device)
    robot = env.unwrapped.scene["robot"]
    terrain_origins = env.unwrapped.scene.env_origins

    with torch.inference_mode():
        while done_count < episodes:
            stepped = env.step(policy(obs))
            obs, dones = stepped[0], stepped[2]
            steps += 1
            cmd = cmd_term.command.torch if hasattr(cmd_term.command, "torch") else cmd_term.command
            err_sum += torch.linalg.norm(cmd[:, :2] - _t(robot.data.root_lin_vel_b)[:, :2], dim=-1)
            yaw_err_sum += torch.abs(cmd[:, 2] - _t(robot.data.root_ang_vel_b)[:, 2])
            root_z = _t(robot.data.root_pos_w)[:, 2]
            height_sum += root_z - _t(terrain_origins)[:, 2]

            finished = dones.nonzero(as_tuple=False).flatten()
            for i in finished.tolist():
                # A time-out means it survived the whole episode; anything else is a fall.
                fell = steps[i].item() < max_len - 1
                lengths.append(steps[i].item())
                fells.append(float(fell))
                mean_xy = (err_sum[i] / steps[i]).item()
                mean_yaw = (yaw_err_sum[i] / steps[i]).item()
                errs.append(mean_xy)
                tracked.append(float(mean_xy < thr_xy and mean_yaw < thr_yaw))
                heights.append((height_sum[i] / steps[i]).item())
                steps[i] = 0.0
                err_sum[i] = 0.0
                yaw_err_sum[i] = 0.0
                height_sum[i] = 0.0
                done_count += 1

    n = len(lengths)
    fall_rate = sum(fells) / n
    return {
        "iteration": iteration,
        "episodes": n,
        # Matches the training log's ``Metrics/success_rate``: tracking quality, not survival.
        "success_rate": sum(tracked) / n,
        "fall_rate": fall_rate,
        "ep_len": sum(lengths) / n,
        "error_vel_xy": sum(errs) / n,
        "pelvis_height": sum(heights) / n,
    }


if __name__ == "__main__":
    main()
