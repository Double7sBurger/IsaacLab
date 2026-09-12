# WBC-AGILE G1 velocity task — port notes

Source: `~/workspace/WBC-AGILE` @ `6830cf9` (v1.3.1, 2026-08-31, GitHub mirror of
`gitlab-master:ml_nav/agile`). Target: IsaacLab `origin/develop` @ `a8b4da3c29` (3.0.0).

The GitLab `main` could not be fetched (302 → SSO login). If it is obtained later, diff it against
the mirror before trusting these files.

## What was copied

Verbatim from `agile/rl_env`, with only the import paths rewritten
(`agile.rl_env.*` → `isaaclab_tasks.contrib.wbc_velocity.*`):

| here | upstream |
| --- | --- |
| `mdp/` | `agile/rl_env/mdp/` (minus the motion-tracking commands/rewards/observations) |
| `utils/` | `agile/rl_env/utils/` |
| `rsl_rl/`, `rsl_rl_observations.py` | `agile/rl_env/rsl_rl/` |
| `assets/robots/unitree_g1.py` | same |
| `velocity_env_cfg.py`, `velocity_history_env_cfg.py`, `agents/` | `agile/rl_env/tasks/locomotion/g1/` |
| `isaaclab_extras/` | `agile/isaaclab_extras/monkey_patches/` (2 of 4, see below) |
| `third_party/rsl_rl/patches/` | same |
| `../../../../../scripts/reinforcement_learning/rsl_rl_wbc/` | `scripts/train.py`, `scripts/cli_args.py` |

Registered ids: `Velocity-G1-WBC-Teacher-v0`, `Velocity-G1-WBC-History-v0`.

## Port deltas (3.0.0b2 → 3.0.0)

Every deviation from upstream, each marked `PORT DELTA` in the code:

1. **`velocity_env_cfg.py::__post_init__` — `sim.use_newton_actuators = False`.**
   Newton-native actuator execution became the default and rejects `DelayedDCMotorCfg`. WBC spawns
   the PhysX variant of the G1 USD and runs a Python-side actuator model, so the Lab actuator path
   is the faithful one.
2. **`mdp/actuators/actuators_cfg.py::DelayedDCMotorCfg.__post_init__` — mirror
   `velocity_limit_sim` onto `actuator_velocity_limit`.**
   `velocity_limit_sim` used to seed the actuator model's own velocity limit, which the DC-motor
   torque-speed curve needs; it now maps only to the solver limit and `ActuatorCollection` clears
   the alias before the actuator is built. WBC's G1 sets only `velocity_limit_sim`.
   `DelayedDCMotor.__init__` additionally drops its recomputation of `_saturation_effort` /
   `_vel_at_effort_lim` — `DCMotor.__init__` now does exactly that itself.
3. **`mdp/__init__.py` — `isaaclab_tasks.manager_based.locomotion.velocity.mdp` →
   `isaaclab_tasks.core.velocity.mdp`** (package layout rename).

## Deliberate omissions

* `agile.isaaclab_extras.monkey_patches.physx_articulation_com_cache` — a 3.0.0b2-only PhysX COM
  cache fix.
* `agile.isaaclab_extras.monkey_patches.terrain_importer_plane_patch` — guarded by
  `_TARGET_ISAACLAB_VERSION == "3.0.0b2"`, and only touches the `eval()` plane.
* `contact_sensor_patch` / `contact_sensor_data_patch` — present upstream but never imported by
  `monkey_patches/__init__.py`.
* Motion-tracking commands/rewards/observations — they need `agile.common.motion_data`, which the
  velocity task does not use.
* `EfficientRecordVideo` → plain `gym.wrappers.RecordVideo` in the train script.

`observation_manager_patch` **is** required: AGILE's RSL-RL wrapper expects non-concatenated
observation groups as a `TensorDict`, and `ObservationManager.compute_group` still returns a plain
dict in 3.0.0.

## Environment

`rsl-rl-lib` 5.4.1 matches the version AGILE pins, and AGILE's 923-line patch is **already applied**
to `env_isaaclab`'s `site-packages/rsl_rl` — `bootstrap.ensure_rsl_rl_patch()` is a no-op here.

This worktree is not pip-installed. Run with its `source/` dirs prepended:

```bash
conda activate env_isaaclab
cd ~/workspace/IsaacLab-wbc
export PYTHONPATH=$(ls -d $PWD/source/*/ | tr '\n' ':')$PYTHONPATH
HEADLESS=1 python scripts/reinforcement_learning/rsl_rl_wbc/train.py \
    --task Velocity-G1-WBC-Teacher-v0 --num_envs 4096
```

## Verified

* Env builds and steps: 29-DoF G1, 12 leg actions, 3 action terms (`joint_pos`,
  `random_upper_body_pos`, `harness`), policy group 7 terms / critic group 9 terms incl. the
  70-value two-foot height scan, 22 rewards, 7 curricula.
* 3 PPO iterations with symmetry data augmentation at 64 envs; all reward, curriculum and
  termination terms log.
