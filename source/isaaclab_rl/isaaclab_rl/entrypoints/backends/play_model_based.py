# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a model-based actor checkpoint in the saved physical task."""

import argparse
import time
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import update_from_dict

from isaaclab_tasks.contrib.motion_tracking.adapter import MotionTrackingAdapter
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from ...model_based.config import REFERENCE_REVISION
from ...model_based.runner import build_actor
from ..common import (
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
)


def run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_play_args(
        parser, agent_default="model_based_cfg_entry_point", agent_help="Saved model-based configuration."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion_dir")
    parser.add_argument("--asset_path")
    parser.add_argument("--kinematics_path")
    parser.add_argument("--steps", type=int, default=1000)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if overrides or args.train_env_cfg or args.frontend != "torch":
        parser.error("Playback restores the saved configuration; use explicit relocation/device/seed options.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["reference_revision"] != REFERENCE_REVISION:
        parser.error("The checkpoint uses a different reference algorithm revision.")
    task = checkpoint["task"]
    if args.task is not None and args.task != task:
        parser.error("The task must match the saved checkpoint.")
    cfg, agent = resolve_task_config(task, checkpoint["agent"], overrides=[])
    update_from_dict(cfg, checkpoint["env_cfg"])
    update_from_dict(agent, checkpoint["agent_cfg"])
    apply_env_overrides(args, cfg)
    cfg.play_mode()
    if args.motion_dir:
        cfg.commands.motion.motion_dir = args.motion_dir
    if args.asset_path:
        cfg.scene.robot.spawn.usd_path = args.asset_path
    if args.kinematics_path:
        cfg.commands.motion.kinematics_path = args.kinematics_path
    output = str(Path(args.checkpoint).parent / "play")
    enable_cameras_for_video(args)
    pre_launch_video_config(cfg, args)
    apply_video_recording(cfg, output, args)
    with launch_simulation(cfg, args):
        env = create_isaaclab_env(task, cfg, args, convert_marl_to_single_agent=False)
        try:
            adapter = MotionTrackingAdapter(env, agent)
            actor = build_actor(adapter, agent).eval()
            actor.load_state_dict(checkpoint["actor"])
            adapter.reset()
            for _ in range(args.steps):
                start = time.monotonic()
                with torch.inference_mode():
                    adapter.advance(actor(adapter.actor_input()))
                if args.real_time:
                    time.sleep(max(0.0, env.unwrapped.step_dt - (time.monotonic() - start)))
        finally:
            env.close()
