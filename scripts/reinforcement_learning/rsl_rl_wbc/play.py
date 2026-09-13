# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Play a trained WBC-AGILE G1 velocity policy, in the viewer or to an mp4.

WBC's own ``scripts/play.py`` feeds sinusoidal joint targets to validate a scene; it does not load a
policy. This does.

Two things differ from training and both matter:

* The harness is faded out by a *curriculum*. With curricula disabled it would come back at full
  strength and hold the robot up, so the action is deleted -- the same thing WBC's ``eval()`` does.
* ``observations.policy`` stays non-concatenated. WBC's ``eval()`` sets ``concatenate_terms=True``,
  which is right for their exported TorchScript but wrong for a checkpoint trained on the dict form.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Velocity-G1-WBC-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--terrain", choices=["rough", "plane"], default="rough",
                    help="'rough' is the terrain it trained on; 'plane' is the measurement condition.")
parser.add_argument("--lin_vel_x", type=float, default=None,
                    help="Pin the forward command [m/s]. Omitted: the task samples its own commands.")
parser.add_argument("--video", action="store_true", help="Record an mp4 instead of opening a viewer.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded length [control steps].")
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps (0 = run forever).")

import cli_args  # isort: skip  # noqa: E402  -- must not import isaaclab_tasks before the app starts

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required: pass a model_*.pt to play")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks.contrib.wbc_velocity.isaaclab_extras  # noqa: F401, E402
import isaaclab_tasks.contrib.wbc_velocity  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv, VideoRecorderCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from isaaclab_tasks.contrib.wbc_velocity.rsl_rl import (  # isort: skip  # noqa: E402
    RslRlVecEnvWrapper,
    make_rsl_rl_runner,
)


def prepare_for_playing(env_cfg):
    env_cfg.curriculum = None
    env_cfg.observations.policy.enable_corruption = False

    if args_cli.terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    elif env_cfg.scene.terrain.terrain_generator is not None:
        # Without the terrain-level curriculum the generator would otherwise stay at level 0.
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # See the module docstring: the harness is curriculum-removed, so it must be deleted by hand.
    if hasattr(env_cfg.actions, "harness"):
        del env_cfg.actions.harness

    if args_cli.lin_vel_x is not None:
        cmd = env_cfg.commands.base_velocity
        cmd.rel_standing_envs = 0.0
        cmd.heading_command = False
        cmd.ranges.lin_vel_x = (args_cli.lin_vel_x, args_cli.lin_vel_x)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
    return env_cfg


def main():
    env_cfg = prepare_for_playing(parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs))
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.video:
        # gym's RecordVideo wrapper is deprecated in 3.0 and its ``render()`` returns None, so a
        # clip recorded through it comes out empty. The recorder also needs a visualizer declared
        # on the sim cfg -- ``--visualizer kit`` alone leaves nothing for it to capture from.
        from isaaclab_visualizers.kit import KitVisualizerCfg

        video_folder = os.path.abspath(os.path.join("logs", "videos", "play"))
        env_cfg.sim.visualizer_cfgs = [
            KitVisualizerCfg(
                # Chase the robot: the task's own viewer sits 25 m back, which is right for
                # watching 4096 environments and useless for watching one gait.
                origin_type="asset",
                origin_track_path="robot",
                origin_env_index=0,
                eye=(-2.5, -3.0, 1.6),
                lookat=(0.0, 0.0, 0.7),
                create_viewport=True,
            )
        ]
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=video_folder,
                output_filename_prefix="wbc_g1",
                video_length=args_cli.video_length,
                # The Kit viewport needs a few frames before it renders anything but grey.
                step_offset=30,
            )
        ]
        print(f"[INFO] recording {args_cli.video_length} steps to {video_folder}")

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = make_rsl_rl_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] loaded {args_cli.checkpoint}")

    # A recording has to outlive the wrapper's own window, or the file is closed mid-write.
    limit = args_cli.steps or (args_cli.video_length + 60 if args_cli.video else 0)

    step = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        while simulation_app.is_running():
            obs, _, _, _ = env.step(policy(obs))
            step += 1
            if limit and step >= limit:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
