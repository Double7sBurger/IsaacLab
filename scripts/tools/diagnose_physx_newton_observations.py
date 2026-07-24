"""Capture PhysX policy states and compare their observations in Newton.

Run this tool twice, from separate processes:

    ./isaaclab.sh -p scripts/tools/diagnose_physx_newton_observations.py \
        --mode capture --task Isaac-Velocity-Rough-AnymalD-PhysxOrder-Compare-Play \
        --checkpoint /path/to/model.pt --num_steps 200 physics=physx

    ./isaaclab.sh -p scripts/tools/diagnose_physx_newton_observations.py \
        --mode replay --task Isaac-Velocity-Rough-AnymalD-PhysxOrder-Compare-Play \
        --snapshot /tmp/physx_states.pt physics=newton_mjwarp

The replay does not advance Newton physics.  For every captured PhysX state it
sets the root pose/velocity, joint position/velocity, previous action, and task
commands in Newton, then recomputes the policy observation.  This isolates
state-to-observation differences from trajectory divergence.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as rsl_cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description="Compare PhysX and Newton observations for identical saved states.")
parser.add_argument("--mode", choices=("capture", "replay"), required=True)
parser.add_argument("--snapshot", type=Path, default=Path("/tmp/physx_newton_observations.pt"))
parser.add_argument("--num_steps", type=int, default=200, help="Policy steps to capture.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to create (default: 1; use one for a deterministic state comparison).",
)
parser.add_argument("--env_id", type=int, default=0, help="Environment index to capture; use one environment for clearest output.")
parser.add_argument("--asset_name", default="robot", help="Articulation whose state is transferred.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic terrain and environment initialization.")
parser.add_argument(
    "--summary_only",
    action="store_true",
    help="For replay, suppress per-step tensors and print only aggregate observation-difference statistics.",
)
parser.add_argument(
    "--outlier_threshold",
    type=float,
    default=1.0e-5,
    help="Absolute-difference threshold used to count replay outlier steps.",
)
rsl_cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _format(tensor: torch.Tensor) -> str:
    """Format one observation/state row without truncating its dimensions."""
    return str(tensor.detach().cpu().numpy())


def _policy_terms(env) -> dict[str, torch.Tensor]:
    """Return named, post-processed policy terms without mutating observation history."""
    manager = env.observation_manager
    terms: dict[str, torch.Tensor] = {}
    for name, cfg in zip(manager._group_obs_term_names["policy"], manager._group_obs_term_cfgs["policy"]):
        value = cfg.func(env, **cfg.params).clone()
        if cfg.modifiers is not None:
            for modifier in cfg.modifiers:
                value = modifier.func(value, **modifier.params)
        # Play configurations must disable corruption.  Applying noise here would
        # consume a different RNG stream in the two processes and invalidate this
        # state-to-observation comparison.
        if cfg.noise is not None and env.cfg.observations.policy.enable_corruption:
            raise RuntimeError("Disable observation corruption before using this diagnostic.")
        if cfg.clip is not None:
            value = value.clip(*cfg.clip)
        if cfg.scale is not None:
            value = value * cfg.scale
        terms[name] = value
    return terms


def _select_env_state(value: object, env_id: int) -> object:
    """Move one environment's complete scene state to CPU, retaining batch dimension."""
    if isinstance(value, torch.Tensor):
        return _cpu(value[env_id : env_id + 1])
    if isinstance(value, dict):
        return {key: _select_env_state(item, env_id) for key, item in value.items()}
    raise TypeError(f"Unexpected scene-state value: {type(value)!r}")


def _capture_state(env, env_id: int) -> dict[str, object]:
    commands = {
        name: _cpu(env.command_manager.get_term(name).command[env_id])
        for name in env.command_manager.active_terms
    }
    return {
        # get_state/reset_to is backend-independent and includes dynamic scene
        # entities, rather than assuming that only the robot affects an obs.
        # Static terrain geometry is recreated from the same task config/seed.
        "scene_state": _select_env_state(env.scene.get_state(is_relative=True), env_id),
        "action": _cpu(env.action_manager.action[env_id]),
        "previous_action": _cpu(env.action_manager.prev_action[env_id]),
        "commands": commands,
    }


def _restore_state(env, state: dict[str, object], env_id: int) -> None:
    """Write a captured articulation/task state and refresh Isaac Lab buffers."""
    ids = torch.tensor([env_id], device=env.device, dtype=torch.long)
    env.reset_to(state["scene_state"], env_ids=ids, is_relative=True)

    env.action_manager._action[env_id] = state["action"].to(env.device)
    env.action_manager._prev_action[env_id] = state["previous_action"].to(env.device)
    for name, command in state["commands"].items():
        env.command_manager.get_term(name).command[env_id] = command.to(env.device)

    # Ray-casters and other non-RTX sensors derive their data from the restored
    # articulation state during scene.update().
    env.scene.update(dt=env.physics_dt)


def _print_capture(
    step: int, state: dict[str, object], policy_observation: torch.Tensor, terms: dict[str, torch.Tensor], env_id: int
):
    print(f"\n[CAPTURE step={step}] state={state}")
    print(f"[CAPTURE step={step}] policy_obs={_format(policy_observation[env_id])}")
    for name, value in terms.items():
        print(f"[CAPTURE step={step}] policy.{name}={_format(value[env_id])}")


def _print_replay(step: int, reference: torch.Tensor, actual: torch.Tensor, reference_terms, actual_terms, env_id: int):
    diff = actual - reference
    print(f"\n[REPLAY step={step}] physx_policy_obs={_format(reference)}")
    print(f"[REPLAY step={step}] newton_policy_obs={_format(actual)}")
    print(
        f"[REPLAY step={step}] abs_diff_max={diff.abs().max().item():.8g} "
        f"abs_diff_mean={diff.abs().mean().item():.8g}"
    )
    for name, value in actual_terms.items():
        term_diff = value[env_id].cpu() - reference_terms[name]
        print(f"[REPLAY step={step}] newton.policy.{name}={_format(value[env_id])}")
        print(f"[REPLAY step={step}] policy.{name}.abs_diff_max={term_diff.abs().max().item():.8g}")


def _print_replay_summary(
    full_stats: list[tuple[int, float, float]], term_stats: dict[str, list[tuple[int, float, float]]], threshold: float
) -> None:
    """Print aggregate max/mean errors and the largest replay outliers."""

    def print_stats(label: str, stats: list[tuple[int, float, float]]) -> None:
        max_step, max_error, _ = max(stats, key=lambda item: item[1])
        mean_error = sum(item[2] for item in stats) / len(stats)
        outliers = [item for item in stats if item[1] > threshold]
        top_steps = sorted(stats, key=lambda item: item[1], reverse=True)[:5]
        top_steps_str = ", ".join(f"{step}:{error:.6g}" for step, error, _ in top_steps)
        print(
            f"[SUMMARY] {label}: max={max_error:.8g} at step={max_step}; "
            f"mean={mean_error:.8g}; outlier_steps={len(outliers)}/{len(stats)} "
            f"(threshold={threshold:g}); top5=[{top_steps_str}]"
        )

    print("\n=== PhysX-to-Newton observation replay summary ===")
    print_stats("policy_obs", full_stats)
    for name, stats in term_stats.items():
        print_stats(f"policy.{name}", stats)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_id >= env_cfg.scene.num_envs:
        raise ValueError(f"--env_id {args_cli.env_id} requires --num_envs > {args_cli.env_id}.")

    # The task's Compare-Play configuration supplies fixed reset state/command
    # and disables noise.  Its PhysX ordering is also required for a PhysX ckpt.
    if getattr(env_cfg.observations.policy, "enable_corruption", True):
        raise ValueError("Use a *-Play task or otherwise set observations.policy.enable_corruption=False.")

    if args_cli.mode == "capture" and not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for --mode capture.")

    agent_cfg = rsl_cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    with launch_simulation(env_cfg, args_cli):
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        base_env = gym.make(args_cli.task, cfg=env_cfg)
        try:
            if args_cli.mode == "capture":
                wrapped_env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
                if agent_cfg.class_name == "OnPolicyRunner":
                    runner_cls = OnPolicyRunner
                elif agent_cfg.class_name == "DistillationRunner":
                    runner_cls = DistillationRunner
                else:
                    raise ValueError(f"Unsupported RSL-RL runner: {agent_cfg.class_name}")
                runner = runner_cls(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
                runner.load(args_cli.checkpoint)
                policy = runner.get_inference_policy(device=base_env.unwrapped.device)

                observation = wrapped_env.get_observations()
                snapshots = []
                for step in range(args_cli.num_steps):
                    raw_env = base_env.unwrapped
                    state = _capture_state(raw_env, args_cli.env_id)
                    terms = _policy_terms(raw_env)
                    policy_observation = observation["policy"]
                    _print_capture(step, state, policy_observation, terms, args_cli.env_id)
                    snapshots.append(
                        {
                            "state": state,
                            "policy_obs": _cpu(policy_observation[args_cli.env_id]),
                            "policy_terms": {name: _cpu(value[args_cli.env_id]) for name, value in terms.items()},
                        }
                    )
                    with torch.inference_mode():
                        action = policy(observation)
                        observation, _, dones, _ = wrapped_env.step(action)
                    if bool(dones[args_cli.env_id]):
                        print(f"[CAPTURE] Environment {args_cli.env_id} reset at step {step}; stopping capture.")
                        break
                args_cli.snapshot.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"asset_name": args_cli.asset_name, "env_id": args_cli.env_id, "samples": snapshots}, args_cli.snapshot)
                print(f"\nSaved {len(snapshots)} PhysX state/observation pairs to {args_cli.snapshot}")
            else:
                data = torch.load(args_cli.snapshot, map_location="cpu", weights_only=True)
                env_id = data["env_id"]
                base_env.reset()
                full_stats: list[tuple[int, float, float]] = []
                term_stats = {name: [] for name in data["samples"][0]["policy_terms"]}
                for step, sample in enumerate(data["samples"]):
                    raw_env = base_env.unwrapped
                    _restore_state(raw_env, sample["state"], env_id)
                    observation = raw_env.observation_manager.compute_group("policy", update_history=False)
                    terms = _policy_terms(raw_env)
                    reference = sample["policy_obs"]
                    actual = observation[env_id].cpu()
                    diff = (actual - reference).abs()
                    full_stats.append((step, diff.max().item(), diff.mean().item()))
                    for name, value in terms.items():
                        term_diff = (value[env_id].cpu() - sample["policy_terms"][name]).abs()
                        term_stats[name].append((step, term_diff.max().item(), term_diff.mean().item()))
                    if not args_cli.summary_only:
                        _print_replay(step, reference, actual, sample["policy_terms"], terms, env_id)
                _print_replay_summary(full_stats, term_stats, args_cli.outlier_threshold)
        finally:
            base_env.close()


if __name__ == "__main__":
    main()
