"""Compare PhysX and MJWarp articulation name ordering for an Isaac Lab task.

Examples:
    ./isaaclab.sh -p scripts/tools/check_articulation_ordering.py \
        --task Isaac-Velocity-Rough-AnymalD --headless physics=physx

    ./isaaclab.sh -p scripts/tools/check_articulation_ordering.py \
        --task Isaac-Velocity-Rough-AnymalD --headless physics=newton_mjwarp
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import get_articulation_name_ordering
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Compare PhysX and MJWarp articulation ordering for one task.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID containing the robot to inspect.")
parser.add_argument("--asset_name", type=str, default="robot", help="Scene articulation name to inspect.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create (default: 1).")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point required to resolve the task configuration.",
)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args


def _print_ordering(kind: str, public_names: tuple[str, ...], physx_names: tuple[str, ...], mjwarp_names: tuple[str, ...]):
    """Print an ordering comparison and the PhysX-to-MJWarp index permutation."""
    print(f"\n{kind.upper()} ORDERING")
    print(f"  public ({len(public_names)}): {list(public_names)}")
    print(f"  physx  ({len(physx_names)}): {list(physx_names)}")
    print(f"  mjwarp ({len(mjwarp_names)}): {list(mjwarp_names)}")

    same_name_set = set(physx_names) == set(mjwarp_names)
    same_order = physx_names == mjwarp_names
    print(f"  same names: {same_name_set}")
    print(f"  same order: {same_order}")
    if same_name_set:
        physx_to_mjwarp = [mjwarp_names.index(name) for name in physx_names]
        print(f"  PhysX index -> MJWarp index: {physx_to_mjwarp}")
    print(f"  public matches PhysX:  {public_names == physx_names}")
    print(f"  public matches MJWarp: {public_names == mjwarp_names}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, _agent_cfg):
    """Instantiate one environment and compare public/backend name conventions."""
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            robot = env.unwrapped.scene[args_cli.asset_name]
            public_joint_names = tuple(robot.joint_names)
            public_body_names = tuple(robot.body_names)
            physx_joint_names = get_articulation_name_ordering(robot, "physx", kind="joint")
            mjwarp_joint_names = get_articulation_name_ordering(robot, "mjwarp", kind="joint")
            physx_body_names = get_articulation_name_ordering(robot, "physx", kind="body")
            mjwarp_body_names = get_articulation_name_ordering(robot, "mjwarp", kind="body")

            print(f"\nActive backend joint order: {list(robot.backend_joint_names)}")
            print(f"Active backend body order:  {list(robot.backend_body_names)}")
            _print_ordering("joint", public_joint_names, physx_joint_names, mjwarp_joint_names)
            _print_ordering("body", public_body_names, physx_body_names, mjwarp_body_names)

            if physx_joint_names == mjwarp_joint_names and physx_body_names == mjwarp_body_names:
                print("\nRESULT: PhysX and MJWarp orders are fully aligned for this asset.")
            else:
                print(
                    "\nRESULT: Ordering differs. A PhysX-trained policy evaluated in Newton should use "
                    'joint_ordering="physx" and body_ordering="physx" to preserve public axes.'
                )
        finally:
            env.close()


if __name__ == "__main__":
    main()
