# G1 motion tracking

`Isaac-Motion-Tracking-G1` trains a PPO tracker. `Isaac-Motion-Tracking-G1-SuperMimic`
trains a reference-conditioned actor through a learned generalized-coordinate world
model, using RSL-RL's `OffPolicyRunner` and `SuperMimic` algorithm.
Both tasks use the same G1 USD, joint configuration, motor parameters, default pose,
action scales and 200 Hz physics / 50 Hz control rate.

## Dependencies and training

This task is stacked on IsaacLab #8878 (motion tracking) and #8867 (SuperMimic
integration). It requires the companion RSL-RL motion-tracking changes on top of
leggedrobotics/rsl_rl#255. The released RSL-RL package in the current lockfile does
not contain these unreleased interfaces. For development, install that checkout
into the Isaac Lab environment:

```bash
uv sync --frozen
uv pip install --no-deps --editable /path/to/rsl_rl-supermimic-motion-tracking
uv run --no-sync isaaclab train \
  --task Isaac-Motion-Tracking-G1-SuperMimic \
  --num_envs 4096 --max_iterations 320000 \
  env.commands.motion.motion_file=/path/to/walk.npz \
  'env.commands.motion.additional_motion_files=[/path/to/dance.npz]'
```

Use `--no-sync` while testing the unreleased learner; an ordinary frozen sync
restores the locked release. Motion files use the named `MotionLoader` NPZ format,
including joint/body names, and must be sampled at 50 Hz. The default single
reference is inherited from the ordinary tracking task. Additional clips are
concatenated in memory, with explicit clip boundaries. Each environment samples
its own start frame; its future reference window never crosses into another clip.
This loader stores body trajectories and is not a streamed large-corpus loader.

## Learner and task responsibilities

Isaac Lab owns the robot, simulation, references, noisy proprioception, context
randomization, reward and soft termination. RSL-RL owns replay, network parameters,
world-model fitting and actor/critic optimization. The actor update is BPTT through
imagined rollouts, not PPO. A pre-reset physical successor is saved on every step.

The model predicts changes in root linear/angular velocity and joint velocity.
The task analytically integrates pose and joint angles using the physics substep
coefficient, then evaluates body tracking through differentiable FK. Joint frames
are read from the same USD and variants used by the simulator; no separate robot
USD or kinematics XML is required. This FK supports unscaled, meter-authored,
floating revolute-joint articulations and validates the joint graph on startup.

The actor consumes ten proprioceptive frames and a ten-frame reference window
spaced five control steps apart. Its goal encoder has a 32-dimensional tanh latent;
encoder and decoder widths are 2048/1024/512/256 with LayerNorm and GELU. Dynamics
uses three 2048-wide hidden layers; the critic uses 256/128. The configured model
window contains seven transitions, the imagined policy horizon is 32, and each
learner iteration performs ten model updates and one actor update. Collection
accrues 0.4 learner iterations per physical step after ten warmup steps. A CLI
iteration is one physical collection step, not one learner update.

## Reproduction scope

The GenCoord integration, integrated-state loss, body tracking objective, soft
termination and network choices follow HybridRobotics/SuperMimic revision
`7e94b073940b7a19b9a82ff01e191cfb35c4ba91`, with proprioceptive actor history.
Motion reward is computed analytically after predicting the successor state.
The model does not learn reward or termination heads in this configuration.

This integration retains Isaac Lab's Newton simulation and the shared motion
tracking reset perturbations and failure-rate sampler. The upstream repository
currently defaults to return-based reset sampling and also provides additional
representations and mixed-task actors. Replay boundaries, critic minibatch
reshuffling and runner iteration accounting follow RSL-RL. Consequently this is
not a claim of bitwise training equivalence or reproduced paper performance.
Multi-GPU learning remains unsupported by the dependency. Checkpoints from the
standalone upstream implementation require explicit conversion.
