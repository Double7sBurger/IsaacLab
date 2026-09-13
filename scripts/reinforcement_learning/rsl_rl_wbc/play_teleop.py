# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive a trained WBC-AGILE G1 velocity policy from the keyboard, with a free camera.

The camera is freed by default. The task's own ``ViewerCfg`` uses ``origin_type="asset_root"``,
which re-aims the viewport at the robot on **every step** -- so orbiting or zooming by hand appears
to do nothing, because the next step overwrites it. Setting ``origin_type="world"`` stops that and
leaves normal Kit navigation working: left-drag orbit, middle-drag pan, right-drag or wheel zoom.
Pass ``--follow`` to get the chase camera back.

Key bindings
------------

======================  ==========================  ==========================
Command                 Key (+)                     Key (-)
======================  ==========================  ==========================
Forward / back          Up arrow, Numpad 8          Down arrow, Numpad 2
Strafe                  Numpad 4                    Numpad 6
Turn                    Z, Numpad 7                 X, Numpad 9
Base height             R                           F
Reset the robot         L
======================  ==========================  ==========================

Keys are read by Kit, so the **viewport window must have focus** — click it once after the scene
loads. Commands are momentary: the robot stops when no key is held.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="Velocity-G1-WBC-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--terrain", choices=["rough", "plane"], default="rough")
parser.add_argument("--v_x", type=float, default=0.8, help="Forward command per key press [m/s].")
parser.add_argument("--v_y", type=float, default=0.4, help="Lateral command per key press [m/s].")
parser.add_argument("--omega_z", type=float, default=1.0, help="Yaw command per key press [rad/s].")
parser.add_argument("--height_step", type=float, default=0.01, help="Base-height change per R/F press [m].")
parser.add_argument("--follow", action="store_true",
                    help="Keep the task's chase camera. It re-aims every step, so the mouse cannot move the view.")

import cli_args  # isort: skip  # noqa: E402  -- must not import isaaclab_tasks before the app starts

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required: pass a model_*.pt to drive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

# Se2Keyboard calls ``omni.appwindow.get_default_app_window()`` without importing the submodule,
# and the extension providing it is not in the default Isaac Lab experience -- so it has to be
# enabled and imported here, or constructing the device raises ModuleNotFoundError.
import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.appwindow", True)
import carb.input  # noqa: F401, E402  -- Se2Keyboard uses carb.input without importing it either
import omni.appwindow  # noqa: F401, E402

import isaaclab_tasks.contrib.wbc_velocity.isaaclab_extras  # noqa: F401, E402
import isaaclab_tasks.contrib.wbc_velocity  # noqa: F401, E402
from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from isaaclab_tasks.contrib.wbc_velocity.rsl_rl import (  # isort: skip  # noqa: E402
    RslRlVecEnvWrapper,
    make_rsl_rl_runner,
)

# The command's own floor for walking; below it the task treats the robot as squatting.
MIN_HEIGHT, MAX_HEIGHT = 0.50, 0.78


def prepare_for_teleop(env_cfg):
    env_cfg.curriculum = None
    env_cfg.observations.policy.enable_corruption = False

    if args_cli.terrain == "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    elif env_cfg.scene.terrain.terrain_generator is not None:
        # Without the terrain-level curriculum the generator would otherwise stay at level 0.
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # The harness is removed by curriculum during training; with curricula off it would return at
    # full strength and hold the pelvis up. WBC's own eval() deletes it too.
    if hasattr(env_cfg.actions, "harness"):
        del env_cfg.actions.harness

    # The command is driven by hand, so nothing else may write it: no resampling, no heading
    # controller stealing the yaw axis, no standing environments zeroing it.
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0

    # Long episodes: a time-out in the middle of driving is only an interruption.
    env_cfg.episode_length_s = 1.0e6

    if not args_cli.follow:
        # See the module docstring: the task's chase camera re-aims every step and silently undoes
        # every mouse orbit and wheel zoom.
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.eye = (3.0, -3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.7)
    return env_cfg


def main():
    env_cfg = prepare_for_teleop(parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs))
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = make_rsl_rl_runner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] loaded {args_cli.checkpoint}")

    command = env.unwrapped.command_manager.get_term("base_velocity")
    device = env.unwrapped.device

    keyboard = Se2Keyboard(
        Se2KeyboardCfg(
            v_x_sensitivity=args_cli.v_x,
            v_y_sensitivity=args_cli.v_y,
            omega_z_sensitivity=args_cli.omega_z,
        )
    )

    height = torch.full((env.unwrapped.num_envs,), float(command.cfg.default_height), device=device)

    def raise_base():
        height.add_(args_cli.height_step).clamp_(MIN_HEIGHT, MAX_HEIGHT)
        print(f"[teleop] base height -> {height[0].item():.3f} m")

    def lower_base():
        height.sub_(args_cli.height_step).clamp_(MIN_HEIGHT, MAX_HEIGHT)
        print(f"[teleop] base height -> {height[0].item():.3f} m")

    keyboard.add_callback("R", raise_base)
    keyboard.add_callback("F", lower_base)
    keyboard.add_callback("L", env.unwrapped.reset)
    keyboard.reset()
    print(keyboard)
    if args_cli.follow:
        print("[teleop] --follow: the camera chases the robot and the mouse cannot move it.")
    else:
        print("[teleop] click the viewport to give it focus, then drive. Camera is free: "
              "left-drag orbit, middle-drag pan, wheel zoom.")

    with torch.inference_mode():
        obs, _ = env.reset()
        while simulation_app.is_running():
            twist = torch.as_tensor(keyboard.advance(), dtype=torch.float32, device=device)
            # Write the *filter target*, not vel_command_b: the command term low-pass filters
            # towards this every step, which is how the policy saw commands change in training.
            command.vel_command_target_b[:] = twist
            command.target_height[:] = height
            command.is_standing_env[:] = False

            obs, _, _, _ = env.step(policy(obs))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
