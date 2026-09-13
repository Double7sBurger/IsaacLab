# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab monkey patches WBC-AGILE applies before building the environment.

Only one of the four upstream patches in ``agile.isaaclab_extras.monkey_patches`` is carried over.
``observation_manager_patch`` is required: AGILE's RSL-RL wrapper expects non-concatenated
observation groups as a ``TensorDict``, and ``ObservationManager.compute_group`` still returns a
plain dict in 3.0.0.

The other three are deliberately omitted:

* ``physx_articulation_com_cache`` -- a 3.0.0b2-only PhysX COM cache fix, already in 3.0.0.
* ``terrain_importer_plane_patch`` -- guarded by ``_TARGET_ISAACLAB_VERSION == "3.0.0b2"``, and
  only touches the ``eval()`` plane.
* ``manager_based_rl_env_patch`` -- it replaces ``ManagerBasedRLEnv.step`` wholesale with a
  3.0.0b2-era copy, which silently drops three things 3.0.0's ``step`` gained: the
  ``compute_final_obs`` terminal observation, the visualizer's UI reset request, and advancing
  ``video_recorders`` (so ``--video`` records an empty clip). It exists to add ``pre_sim_step``
  events and a ``_disable_terminations`` flag, and the velocity task uses neither -- both belong to
  the stand-up / fallen-state-dataset tasks, which are not ported here. Anything that does need
  them should re-add the two hooks on top of the current ``step``, not fork it.
"""

from .observation_manager_patch import *  # noqa: F401, F403
