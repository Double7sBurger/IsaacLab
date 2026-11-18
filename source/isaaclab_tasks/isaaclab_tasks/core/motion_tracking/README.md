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
window contains nine stored states and eight transitions, the imagined policy horizon is 32, and each
learner iteration performs ten model updates and one actor update. Collection
accrues 0.4 learner iterations per physical step, skipping fits until nine states
are stored. A CLI
iteration is one physical collection step, not one learner update.

Adam starts at 2e-4 and follows a cosine schedule over 250,000 physical steps,
ending at 0.03 times the initial rate. Collection adds action noise with standard
deviation 0.1. CUDA graphs capture learner updates; set
`agent.algorithm.use_cuda_graph=false` for CPU debugging. The default 4096-environment
configuration holds 1000 replay states per environment and performs 127,997 learner
updates over 320,000 physical steps. If changing environment count while preserving
the data ratio, scale replay capacity and `updates_per_step` with it, or scale the
learner batch sizes by the same factor.

## Reproduction scope

The reference implementation is HybridRobotics/SuperMimic revision
`5efbf114e11f725c31d83e217be987c13c0e4ce2`. The training preset follows the
Isaac Lab adaptation used for the BONES-SEED 320k policy validated on G1:
GenCoord, proprioceptive history, pure Adam, eight predicted transitions,
action noise and physical-step cosine decay. These are explicit choices from
that run, not the unmodified upstream defaults (which include Muon/Adam).
Motion reward is computed analytically after predicting the successor state.
The model does not learn reward or termination heads in this configuration.

Complete replay windows are sampled uniformly across eligible environments,
then across valid starts within each environment. Critic targets use uncapped
bootstrap values before the final return cap, uniform regression weights and
one minibatch permutation reused across five epochs. Reset sampling uses measured
returns with the reference explore/focus/keep mixture. Context includes absolute
local torso CoM, foot friction and encoder bias. Resets retain the sampled frame
for the first action, with the same pose and velocity perturbation ranges.

Isaac Lab owns the shared motion-tracking USD and Newton simulation. This is not
a claim of bitwise simulator equivalence or reproduced paper performance. The
validated run used two L40 GPUs with 2048 environments per rank; this preset uses
the same global batch sizes on one device. Multi-GPU learning and rank-local
motion sharding remain unsupported by the RSL-RL dependency. Additional upstream
representations and mixed-task actors are outside this task. Standalone checkpoints
require explicit conversion.
