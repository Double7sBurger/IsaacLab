# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task contracts consumed by the model-based learners.

Tasks own state integration, differentiable objectives, and imagined environments.
The learners do not depend on a particular robot, simulator, or observation layout.
"""

from typing import Protocol

import torch
from tensordict import TensorDict


class TaskSpec(Protocol):
    """Dimensions and routing name of one goal encoder."""

    name: str
    goal_dim: int


class StateRepresentation(Protocol):
    """State conversion and integration supplied by a task's torch module."""

    rep_state_dim: int
    prediction_dim: int
    dt: float
    LOSS_WEIGHT_CONFIG: dict[str, str]

    def represent(self, full_state: torch.Tensor) -> torch.Tensor: ...

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor: ...

    def apply_prediction(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor, prediction: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]: ...


class WorldModelEnv(Protocol):
    """Differentiable rollout environment, with no simulator dependency."""

    dynamics: torch.nn.Module
    representation: StateRepresentation
    max_step_reward: float

    def sample_initial_states(self) -> TensorDict: ...

    def reset(self, batch: TensorDict) -> TensorDict: ...

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]: ...
