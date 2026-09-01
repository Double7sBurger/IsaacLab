Added
^^^^^

* Added the ``Isaac-Velocity-Rough-G1-DR`` and ``Isaac-Velocity-Flat-G1-DR`` tasks, sim-to-real
  oriented variants of ``Isaac-Velocity-Rough-G1`` and ``Isaac-Velocity-Flat-G1``. The actor observes
  only quantities a physical G1 can stream (IMU angular velocity and gravity direction, joint
  encoders, velocity command, last action), each with per-step noise and a per-episode bias, stacked
  over five frames; base linear velocity and the terrain height scan move to a privileged ``critic``
  observation group. Dynamics are randomized over contact friction, per-link mass, torso payload and
  center of mass, joint armature and friction, and actuator gains, and the leg and foot joint
  commands are lagged by 0-20 ms through :class:`~isaaclab.actuators.DelayedPDActuator`.
