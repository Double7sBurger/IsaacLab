Added
^^^^^

* Added the 29-DoF sim-to-real task family for the G1, built on the robot description Unitree ships
  today rather than the superseded 37-joint one: ``Isaac-Velocity-Flat-G1-DR29``,
  ``Isaac-Velocity-Rough-G1-DR29``, and three rough variants that differ only in how base height is
  constrained -- ``-DR29-Official`` (a 0.4 m floor above the terrain), ``-DR29-OfficialReward``
  (a reward toward 0.72 m, no floor) and ``-DR29-OfficialTeacher`` (the floor, plus the terrain
  height scan moved into the actor's observation group so the policy can serve as a distillation
  teacher). The asset is selected with the ``G1_29DOF_USD`` environment variable.
* Added ``scripts/eval_checkpoints.py``, which scores every checkpoint of a run against held-out
  randomization draws with the terrain curriculum disabled, and ``scripts/plot_heldout_eval.py`` to
  chart the result. Both exist because ``Metrics/success_rate`` in the training log is measured on
  the environments the policy is training in, and is blind to posture: over one 10000-iteration run
  it stayed at 0.95 while mean pelvis height fell 25 cm.
