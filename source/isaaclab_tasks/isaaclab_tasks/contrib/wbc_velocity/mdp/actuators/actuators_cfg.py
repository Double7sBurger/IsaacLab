# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.wbc_velocity.mdp.actuators.actuators import DelayedDCMotor, DelayedImplicitActuator


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    """Configuration for delayed direct control (DC) motor actuator model."""

    class_type: type = DelayedDCMotor

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""  # noqa: E501

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""  # noqa: E501

    def __post_init__(self):
        # PORT DELTA (isaaclab 3.0.0b2 -> 3.0.0): ``velocity_limit_sim`` used to seed the actuator
        # model's own velocity limit, which the DC-motor torque-speed curve needs. It now maps only
        # to the solver limit (``joint_velocity_limit``), and ``ActuatorCollection`` clears the
        # alias before the actuator is built. WBC's G1 sets only ``velocity_limit_sim``, so mirror
        # it here to keep the torque-speed curve identical to upstream.
        if self.actuator_velocity_limit is None and self.velocity_limit is None:
            self.actuator_velocity_limit = self.velocity_limit_sim


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedImplicitActuator

    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""
