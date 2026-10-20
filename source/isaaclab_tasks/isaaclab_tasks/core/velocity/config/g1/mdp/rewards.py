# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab.utils.warp import raycast_mesh


def pelvis_height_deficit_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Squared shortfall of the root below ``target_height`` above the ground beneath it."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    clearance = asset.data.root_pos_w.torch[:, 2] - ground
    return torch.clamp(target_height - clearance, min=0.0).square()


def pelvis_below_terrain_clearance(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the root sits less than ``minimum_height`` above the ground beneath it."""
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    return asset.data.root_pos_w.torch[:, 2] - ground < minimum_height


def pelvis_below_terrain_clearance_after_warmup(
    env,
    minimum_height: float,
    warmup_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
):
    """:func:`pelvis_below_terrain_clearance`, inert for the first ``warmup_steps`` steps."""
    if env.common_step_counter < warmup_steps:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return pelvis_below_terrain_clearance(env, minimum_height, asset_cfg, sensor_cfg)


def joint_deviation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared deviation of the selected joints from their default angles."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def body_ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared world-frame roll/pitch angular velocity of a named body."""
    asset = env.scene[asset_cfg.name]
    omega = asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(omega[:, :2]), dim=1)


def root_downward_speed_l2(
    env, speed_limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize world-frame downward root speed exceeding the limit [m/s]."""
    velocity_z = env.scene[asset_cfg.name].data.root_lin_vel_w.torch[:, 2]
    return torch.clamp(-velocity_z - speed_limit, min=0.0).square()


def sole_edge_distances(sole_points, mesh, height_tolerance=0.04, max_distance=0.25):
    """Estimate horizontal sole-corner overhang [m] beyond the supporting terrain plane."""
    starts = sole_points.clone()
    starts[..., 2] += 0.5
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.0
    hits, _, normals, _ = raycast_mesh(starts, directions, mesh, max_dist=2.0, return_normal=True)
    valid = torch.isfinite(hits).all(-1) & torch.isfinite(normals).all(-1)
    support = valid & ((hits[..., 2] - sole_points[..., 2]).abs() < height_tolerance) & (normals[..., 2] > 0.7)
    support_index = torch.where(support, hits[..., 2], -torch.inf).argmax(-1)
    gather = support_index[..., None, None].expand(*support_index.shape, 1, 3)
    anchor = torch.nan_to_num(hits.gather(-2, gather).squeeze(-2), posinf=0.0, neginf=0.0)
    normal = torch.nan_to_num(normals.gather(-2, gather).squeeze(-2), posinf=0.0, neginf=0.0)
    corner_indices = [0, 2, 6, 8]
    corners = sole_points[..., corner_indices, :]
    ground = hits[..., corner_indices, 2]
    plane_z = anchor[..., None, 2] - ((corners[..., :2] - anchor[..., None, :2]) * normal[..., None, :2]).sum(
        -1
    ) / normal[..., None, 2].clamp_min(0.7)
    outside = (plane_z - ground > height_tolerance) | ~valid[..., corner_indices]
    angles = torch.arange(16, device=sole_points.device, dtype=sole_points.dtype) * (2 * math.pi / 16)
    horizontal = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), -1)
    ray_start = corners.clone()
    ray_start[..., 2] = plane_z - 0.01
    ray_start = ray_start[..., None, :].expand(*corners.shape[:-1], 16, 3).contiguous()
    ray_direction = horizontal.expand_as(ray_start).contiguous()
    _, distances, _, _ = raycast_mesh(
        ray_start.reshape(-1, 16, 3),
        ray_direction.reshape(-1, 16, 3),
        mesh,
        max_dist=max_distance,
        return_distance=True,
    )
    distance = distances.amin(-1).reshape(*corners.shape[:-1]).clamp_max(max_distance)
    return torch.where(support.any(-1)[..., None], torch.where(outside, distance, 0.0), max_distance)


def feet_support_edge_distance(env, sensor_cfg, asset_cfg, sole_size, sole_offset) -> torch.Tensor:
    """Penalize mean corner overhang [m] for feet carrying more than 20 N."""
    robot = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]
    positions = robot.data.body_pos_w.torch[:, asset_cfg.body_ids]
    quaternions = robot.data.body_quat_w.torch[:, asset_cfg.body_ids]
    xy = torch.cartesian_prod(
        torch.linspace(-0.5, 0.5, 3, device=env.device), torch.linspace(-0.5, 0.5, 3, device=env.device)
    )
    points = torch.cat((xy, torch.full_like(xy[:, :1], -0.5)), -1)
    points = points * torch.tensor(sole_size, device=env.device) + torch.tensor(sole_offset, device=env.device)
    points = points.expand(*positions.shape[:-1], 9, 3)
    points = quat_apply(quaternions[..., None, :].expand(*points.shape[:-1], 4), points) + positions[..., None, :]
    scanner = env.scene["height_scanner"]
    key = (scanner.cfg.mesh_prim_paths[0], scanner.device)
    if key not in scanner.meshes:
        from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster

        BaseRayCaster._initialize_warp_meshes(scanner)
    mesh = scanner.meshes[key]
    distance = sole_edge_distances(points, mesh).mean(-1)
    contact = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > 20.0
    return (distance * contact).sum(-1)
