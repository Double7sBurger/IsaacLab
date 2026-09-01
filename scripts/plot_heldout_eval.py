# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot the held-out sweeps written by :mod:`eval_checkpoints` for one or more runs.

Two measures with different units share the x-axis but never a y-axis: tracking success is a
fraction, pelvis height is metres. They get one panel each, stacked, so the reader compares each
measure against itself across runs rather than against the other measure's scale.

Example::

    uv run python scripts/plot_heldout_eval.py \\
        "no constraint=logs/rsl_rl/g1_rough_dr29/2026-08-31_14-56-34/eval_heldout.csv" \\
        "floor 0.4=logs/rsl_rl/g1_rough_dr29/2026-08-31_21-57-33/eval_heldout.csv" \\
        --out /tmp/heldout.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Slots 1-3 of the data-viz reference palette, in fixed order. Categorical hues are assigned by
# position and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#dedcd4"

PANELS = [
    ("success_rate", "Held-out success rate", "fraction of episodes tracking the command"),
    ("pelvis_height", "Pelvis height (m)", "mean height above the terrain beneath the robot"),
]


def _read(path: str) -> dict[str, list[float]]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", nargs="+", help='One or more "label=path/to/eval_heldout.csv".')
    parser.add_argument("--out", default="heldout_eval.png")
    args = parser.parse_args()

    runs = []
    for spec in args.series:
        label, _, path = spec.partition("=")
        runs.append((label, _read(path)))

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True, facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.22)

    for ax, (column, ylabel, subtitle) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

        for i, (label, data) in enumerate(runs):
            ax.plot(
                data["iteration"],
                data[column],
                color=SERIES[i % len(SERIES)],
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=label,
                zorder=3 - i,
            )
            # Direct-label the end of each line so identity never rests on colour alone.
            ax.annotate(
                label,
                (data["iteration"][-1], data[column][-1]),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                fontsize=9,
                color=TEXT_SECONDARY,
            )

        ax.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=10)
        ax.set_title(subtitle, color=TEXT_SECONDARY, fontsize=9, loc="left", pad=6)
        ax.set_xlim(left=0)
        ax.margins(x=0.16)

    axes[0].legend(frameon=False, loc="lower right", fontsize=9, labelcolor=TEXT_SECONDARY)
    axes[-1].set_xlabel("training iteration", color=TEXT_PRIMARY, fontsize=10)
    fig.suptitle(
        "G1 rough DR29 — held-out evaluation across checkpoints",
        color=TEXT_PRIMARY,
        fontsize=13,
        x=0.125,
        ha="left",
        y=0.97,
    )
    fig.savefig(args.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
