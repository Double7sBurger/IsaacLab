G1 YMSDM simulation reference
============================

This branch preserves the configuration used by the original ``ymsdm_s42`` depth
student. Its teacher is the YMS Newton mesh policy, seed 43; the student is seed 42.
The teacher uses PPO with mirror data augmentation. The original student uses
supervised distillation without mirror augmentation. Later mirror-distilled,
torso-penalty, taller-stair and torque-limit experiments are separate configurations.

Source configuration
--------------------

* `Robot and actuators <../../../source/isaaclab_assets/isaaclab_assets/robots/unitree.py>`_:
  ``G1_29DOF_VELOCITY_CFG``.
* `Teacher environment and action scales <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/rough_29dof_mjlab_env_cfg.py>`_:
  ``G129DofRoughMjlabScaleEnvCfg``. Its parent classes define the terrain,
  randomization, rewards and terminations.
* `Newton physics preset <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py>`_:
  ``RoughPhysicsCfg.newton_mjwarp``.
* `Student environment and camera <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/rough_29dof_depth_distill_env_cfg.py>`_:
  ``G129DofRoughYmsDepthDistillEnvCfg`` and ``_wire_depth_student``.
* `Learning configuration <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/config/g1/agents/rsl_rl_ppo_cfg.py>`_:
  ``G1RoughSymmetryPPORunnerCfg`` and
  ``G129DofRoughAirTime100DepthDistillationRunnerCfg``.
* `Teacher mirror transform <../../../source/isaaclab_tasks/isaaclab_tasks/core/velocity/mdp/symmetry/g1_29dof.py>`_:
  ``compute_symmetric_states``.

Simulation settings
-------------------

.. list-table:: Original recipe
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - Physics
     - Newton MuJoCo-Warp, ``implicitfast``, pyramidal friction cone
   * - Timing
     - Simulation dt 5 ms, two solver substeps (2.5 ms), decimation 4 (50 Hz policy)
   * - Terrain
     - Triangle mesh for all six terrain families, including slopes and rough patches
   * - Contact defaults
     - Margin 0 m, stiffness 160000, damping 1100; Newton collision pipeline
   * - Robot
     - NVIDIA-shipped G1 USD, 29 body joints plus 14 Dex3 finger joints
   * - Foot collision geometry
     - One sole box per foot, approximately 203.1 x 65.5 x 18.5 mm
   * - Actions
     - 43 joint-position offsets; per-joint scales in ``_MJLAB_ACTION_SCALE``
   * - Training
     - 4096 environments; teacher 6000 iterations, student 8000 iterations

The asset is ``{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd`` with the
``g1_a1_feet.usda`` override. This layer replaces the foot colliders; it does not
replace the shipped mass or inertia properties. It is not the older
``g1_minimal.usd`` robot, although the sole dimensions are taken from that asset.

.. list-table:: Original PD gains and simulation effort limits
   :header-rows: 1

   * - Joint group
     - Stiffness [N m/rad]
     - Damping [N m s/rad]
     - Effort limit [N m]
   * - Hip pitch, knee, waist yaw
     - 200
     - 5
     - 300
   * - Hip roll/yaw
     - 150
     - 5
     - 300
   * - Ankle pitch/roll
     - 20
     - 2
     - 20
   * - Waist pitch/roll
     - 200
     - 5
     - 50
   * - Shoulders, elbows, wrists, fingers
     - 40
     - 10
     - 300

Body-joint armature is 0.01 kg m²; finger armature is 0.001 kg m². Action scales
and simulation torque saturation are separate settings: changing one does not
automatically update the other.

The original policy has 43 action outputs, including the fingers. The deployment
robot has 29 body joints; deployment must preserve the training observation layout
and map body commands by joint name, rather than truncating the action vector.
The student reads 690 proprioceptive values (five samples per observation term)
and three 38 x 64 depth images, oldest first. The teacher reads 328 values,
including privileged velocity and terrain heights.

The camera is attached to ``torso_link`` at position
``(0.0576, 0.0175, 0.4299)`` m and quaternion ``(0.9150, 0, -0.4035, 0)``
in wxyz order, using the ``world`` camera convention. Horizontal FOV is 87 degrees.
The student uses image-plane depth clipped to 3 m and divided by 3. The renderer's
near/far clipping range is 0.05--10 m. ``DepthImageStack`` flips both image axes
before stacking; deployment must match the resulting orientation and apply this
conversion exactly once. ``update_latest_camera_pose=True`` keeps the rendered
view attached to the moving torso.

Training commands
-----------------

Use this checkout's Python environment and dependencies. Download both NVIDIA
asset packages with their referenced files intact, then generate the sole layer:

.. code-block:: bash

    uv run python scripts/make_g1_ladder.py \
      --old /path/to/G1_isaaclab/g1_minimal.usd \
      --new /path/to/G1_shipped/g1.usd \
      --out_dir /path/to/g1_layers

Use only ``g1_a1_feet.usda`` from this command. The other generated layers modify
additional properties and are not part of this recipe. Asset binaries and
checkpoints are not included in this repository.

Run the following in one Bash session, replacing the asset path:

.. code-block:: bash

    G1_SOLE_USD=/path/to/g1_layers/g1_a1_feet.usda
    common=(physics=newton_mjwarp "env.scene.robot.spawn.usd_path=$G1_SOLE_USD")
    for terrain in pyramid_stairs pyramid_stairs_inv boxes random_rough \
                   hf_pyramid_slope hf_pyramid_slope_inv; do
      common+=("env.scene.terrain.terrain_generator.sub_terrains.$terrain.convert_to_heightfield=false")
    done

    uv run python scripts/reinforcement_learning/train.py \
      --rl_library rsl_rl \
      --task Isaac-Velocity-Rough-G1-29Dof-AirTime100-MjlabScale-Sym \
      --num_envs 4096 --seed 43 --max_iterations 6000 \
      --run_name yms_mesh_s43 "${common[@]}"

Once teacher training finishes, point to its final checkpoint and start a fresh
student. ``--checkpoint`` supplies the teacher to the distillation runner:

.. code-block:: bash

    TEACHER_CKPT=/path/to/teacher/run/model_5999.pt
    uv run python scripts/reinforcement_learning/train.py \
      --rl_library rsl_rl \
      --task Isaac-Velocity-Rough-G1-29Dof-AirTime100-Yms-DepthDistill \
      --num_envs 4096 --seed 42 --max_iterations 8000 \
      --run_name ymsdm_s42 --checkpoint "$TEACHER_CKPT" "${common[@]}"

The iteration and mesh overrides are intentional: the task defaults alone do not
describe the historical run. Preserve the resolved ``params/env.yaml`` and
``params/agent.yaml`` together with checkpoints from new runs.

What helped with simulation-to-real comparison
----------------------------------------------

Align the joint mapping, default poses, action scales, gains, control period and
torque saturation before attributing differences to the contact solver. In our
deployment comparisons, actuator saturation was a particularly visible mismatch:
the original Newton ankle limit was 20 N m, while the MuJoCo model used 50 N m.
With the same original student, raising only the Newton ankle limit made its sway
look more like MuJoCo and the observed real-robot gait. This increased sway; it
was not a policy improvement or a measurement of the hardware's effective limit.

In one fixed-checkpoint, 20-second Newton mesh evaluation over a 10-degree slope
and rough patches, pelvis roll standard deviation changed from 0.90 to 3.27 degrees
when only the ankle limit changed from 20 to 50 N m. Matching all 29 body limits to
MuJoCo gave 3.32 degrees; changing only the other body limits gave 1.12 degrees.
The command was 0.5 m/s with heading and route-centering feedback; statistics used
seconds 2--20. These are single-route comparisons, not general stability guarantees.
The original limits above remain unchanged in this reference.

Verify effective contact parameters in the backend, not just configuration field
names. In our Newton runtime audit, the foot/ground friction coefficient was 1.0:
the ground's 1.0 dominated the foot's 0.8 under the material combination. The
configured dynamic-friction value of 0.6 did not create a separate sliding-friction
coefficient in this Newton path. Sole geometry, passive joint friction, mass and
inertia also need checking against the deployment model.

For the depth student, match the moving camera pose, image orientation, metric
depth convention, clipping and history order. These are part of the policy's input
contract, not merely visualization settings. The original student worked reasonably
on hardware, but drift on unfamiliar terrain remained; later mirror distillation
addressed that separately. This reference is not a fully identified model of the
real robot.

Provenance
----------

The configuration sources are preserved from commit
``960d7828401d8a275bcd420cb6694f6eb4b13ff8``. The original student is
``ymsdm/s42/model_7999.pt``, SHA-256
``e0d1714d9c6985b8170825a8a5a0b2c17e9ea92d5c168e114907541750465f5b``.
The reference was checked against the historical workflow, checkpoint and exported
deployment contract. The original run's full resolved configuration and binary
environment were not archived together, so these commands describe the recovered
recipe rather than guaranteeing a bitwise reproduction of the September run.
