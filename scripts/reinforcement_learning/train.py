# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified training executable for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from isaaclab_rl.entrypoints import run_train_cli


def _expand_config_args(argv: list[str]) -> list[str]:
    """Expand a YAML training profile into the regular unified-train CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    config_args, remaining = parser.parse_known_args(argv)
    if config_args.config is None:
        return argv

    config_path = config_args.config.expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Training config must contain a mapping: {config_path}")

    unknown_sections = set(config) - {"arguments", "presets"}
    if unknown_sections:
        raise ValueError(f"Unknown training config section(s): {sorted(unknown_sections)}")

    expanded: list[str] = []
    arguments = config.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("The 'arguments' section must be a mapping.")
    for name, value in arguments.items():
        option = f"--{name}"
        if isinstance(value, bool):
            if value:
                expanded.append(option)
        elif value is not None:
            expanded.append(f"{option}={value}")

    presets = config.get("presets", {})
    if not isinstance(presets, dict):
        raise ValueError("The 'presets' section must be a mapping.")
    expanded.extend(f"{name}={value}" for name, value in presets.items() if value is not None)

    # Explicit CLI values come last and therefore can override scalar values
    # from the profile for one-off experiments.
    print(f"[INFO] Loaded training config: {config_path}")
    return expanded + remaining


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning training library."""
    effective_argv = sys.argv[1:] if argv is None else argv
    return run_train_cli(_expand_config_args(effective_argv))


if __name__ == "__main__":
    raise SystemExit(main())
