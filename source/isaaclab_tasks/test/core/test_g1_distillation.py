# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from isaaclab.utils import to_dict

from isaaclab_tasks.core.velocity.config.g1.agents.distillation import MirrorDistillation
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughDepthStudentRunnerCfg


def test_mirror_distillation_load_normalize_and_update():
    """PPO loading starts a new student; each rollout step normalizes both reflections exactly once."""
    torch.manual_seed(42)
    robot = SimpleNamespace(
        joint_names=["left_shoulder_roll_joint", "right_shoulder_roll_joint"],
        data=SimpleNamespace(
            joint_pos_limits=torch.tensor([[[-0.5, 1.0], [-1.0, 0.5]]]),
            default_joint_pos=torch.tensor([[0.16, -0.16]]),
        ),
        find_joints=lambda *args, **kwargs: ([0, 1], ["left_shoulder_roll_joint", "right_shoulder_roll_joint"]),
    )
    env = SimpleNamespace(
        num_envs=2,
        num_actions=2,
        device="cpu",
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            active_terms=["joint_pos"],
            get_term=lambda name: SimpleNamespace(cfg=SimpleNamespace(joint_names=[".*"], preserve_order=False)),
        ),
        observation_manager=SimpleNamespace(
            active_terms={
                "policy": ["base_ang_vel", "joint_pos", "actions"],
                "teacher": ["base_ang_vel", "joint_pos", "actions", "height_scan"],
            },
            group_obs_term_dim={"policy": [(15,), (10,), (10,)], "teacher": [(3,), (2,), (2,), (187,)]},
            cfg=SimpleNamespace(
                **{group: SimpleNamespace(joint_pos=SimpleNamespace(params={})) for group in ("policy", "teacher")}
            ),
        ),
    )
    env.unwrapped = env
    obs = TensorDict(
        {
            "policy": torch.randn(2, 35),
            "teacher": torch.randn(2, 194),
            "depth": torch.rand(2, 3, 38, 64),
            "depth_mirror": torch.rand(2, 3, 38, 64),
        },
        [2],
    )
    cfg = to_dict(G1RoughDepthStudentRunnerCfg())
    cfg["multi_gpu"] = None
    # Architecture width is irrelevant to loading, reflection and normalization semantics.
    cfg["student"]["hidden_dims"] = cfg["teacher"]["hidden_dims"] = [16]
    with pytest.warns(UserWarning, match="not divisible"):
        alg = MirrorDistillation.construct_algorithm(obs, env, cfg, "cpu")
    student_before = {k: v.clone() for k, v in alg.student.state_dict().items()}
    teacher_state = {k: v.clone() + 0.01 for k, v in alg.teacher.state_dict().items()}
    resumed = alg.load(
        {"actor_state_dict": teacher_state, "iter": 5999},
        {"student": True, "teacher": True, "optimizer": False, "iteration": True},
    )
    assert alg.teacher_loaded and not resumed
    assert not alg.optimizer.state
    for name, value in alg.student.state_dict().items():
        torch.testing.assert_close(value, student_before[name])
    mirror = alg.mirror.observations(obs)
    for name, value in alg.mirror.observations(mirror).items():
        torch.testing.assert_close(value, obs[name])
    torch.testing.assert_close(
        mirror["policy"][:, :15].reshape(2, 5, 3), obs["policy"][:, :15].reshape(2, 5, 3) * torch.tensor([-1, 1, -1])
    )
    torch.testing.assert_close(
        mirror["policy"][:, 15:25].reshape(2, 5, 2), -obs["policy"][:, 15:25].reshape(2, 5, 2).flip(-1)
    )
    torch.testing.assert_close(mirror["depth"], obs["depth_mirror"].flip(-1))
    torch.testing.assert_close(
        mirror["teacher"][:, 7:].reshape(2, 11, 17), obs["teacher"][:, 7:].reshape(2, 11, 17).flip(1)
    )
    expected_samples = []
    for step in range(24):
        with torch.no_grad():
            alg.act(obs)
            obs = obs.clone()
            obs["policy"] += 0.02
            expected_samples.extend([alg.mirror.observations(obs)["policy"], obs["policy"]])
            alg.process_env_step(obs, torch.zeros(2), torch.tensor([step % 3 == 0, False]), {})
    normalizer = alg.student.obs_normalizer
    expected = torch.cat(expected_samples)
    torch.testing.assert_close(normalizer._mean, expected.mean(0, keepdim=True))
    torch.testing.assert_close(normalizer._var, expected.var(0, unbiased=False, keepdim=True))
    before_update = {k: v.clone() for k, v in normalizer.state_dict().items()}
    loss = alg.update()["behavior"]
    assert torch.isfinite(torch.tensor(loss)) and loss > 0
    for name, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, before_update[name])
    for name, value in alg.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_state[name])
    assert any(
        not torch.equal(value, student_before[name])
        for name, value in alg.student.state_dict().items()
        if name.startswith("mlp.")
    )


def test_support_edge_distance_on_a_step():
    """A sole hanging 8 cm over a step is charged only at its two unsupported corners."""
    import trimesh

    from isaaclab.utils.warp import convert_to_warp_mesh

    from isaaclab_tasks.core.velocity.config.g1.mdp.rewards import sole_edge_distances

    step = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    step.apply_translation((0.0, 0.0, -0.5))
    mesh = convert_to_warp_mesh(step.vertices, step.faces, device="cpu")
    xy = torch.cartesian_prod(torch.tensor([-0.1, 0.0, 0.1]), torch.tensor([-0.04, 0.0, 0.04]))
    sole = torch.cat((xy, torch.zeros(9, 1)), dim=-1)
    centers = torch.tensor([[0.0, 0.0, 0.0], [0.48, 0.0, 0.0], [2.0, 0.0, 0.0]])
    distances = sole_edge_distances(sole[None] + centers[:, None], mesh)
    torch.testing.assert_close(
        distances,
        torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.08, 0.08], [0.25, 0.25, 0.25, 0.25]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_depth_history_orientation_and_partial_reset():
    """Deployment keeps oldest-to-newest channels and the calibrated 180-degree image rotation."""
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

    from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import DepthImageStack

    pixels = torch.tensor([[[[0.0], [1.0]], [[2.0], [3.0]]]]).repeat(2, 1, 1, 1)
    camera = SimpleNamespace(data=SimpleNamespace(output={"depth": pixels}))
    env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(sensors={"camera": camera}))
    term = DepthImageStack(ObservationTermCfg(func=DepthImageStack), env)
    sensor = SceneEntityCfg("camera")
    rotated = torch.tensor([[[[3.0, 2.0], [1.0, 0.0]]]]).repeat(2, 1, 1, 1)
    first = term(env, sensor, "depth")
    torch.testing.assert_close(first, rotated.repeat(1, 3, 1, 1))
    pixels.add_(4.0)
    second = term(env, sensor, "depth")
    torch.testing.assert_close(second, torch.cat([rotated, rotated, rotated + 4], dim=1))
    term.reset([0])
    pixels.add_(4.0)
    third = term(env, sensor, "depth")
    torch.testing.assert_close(third[0:1], (rotated[0:1] + 8).repeat(1, 3, 1, 1))
    torch.testing.assert_close(third[1:2], torch.cat([rotated[1:2], rotated[1:2] + 4, rotated[1:2] + 8], dim=1))
    torch.testing.assert_close(first, rotated.repeat(1, 3, 1, 1))
