# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab monkey patches WBC-AGILE applies before building the environment.

Ported from ``agile.isaaclab_extras.monkey_patches``. Two of the four upstream patches are
deliberately omitted here:

* ``physx_articulation_com_cache`` -- a 3.0.0b2-only PhysX COM cache fix, already in 3.0.0.
* ``terrain_importer_plane_patch`` -- guarded by ``_TARGET_ISAACLAB_VERSION == "3.0.0b2"`` and
  only affects the ``eval()`` plane, not training.
"""

from .manager_based_rl_env_patch import *  # noqa: F401, F403
from .observation_manager_patch import *  # noqa: F401, F403
