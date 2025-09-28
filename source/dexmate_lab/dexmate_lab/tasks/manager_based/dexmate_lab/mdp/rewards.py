# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance.

    The reward is close to 1 when the maximum distance between the object and any end-effector body is small.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def contacts(env: ManagerBasedRLEnv, threshold: float, binary_contact: bool = True) -> torch.Tensor:
    """Check if grasp contacts are good (thumb + at least one other finger)."""
    
    # Get contact sensors for Vega fingertips
    thumb_contact_sensor: ContactSensor = env.scene.sensors["R_th_l2_contact"]
    ff_contact_sensor: ContactSensor = env.scene.sensors["R_ff_l2_contact"] 
    mf_contact_sensor: ContactSensor = env.scene.sensors["R_mf_l2_contact"] 
    rf_contact_sensor: ContactSensor = env.scene.sensors["R_rf_l2_contact"] 
    lf_contact_sensor: ContactSensor = env.scene.sensors["R_lf_l2_contact"]
    
    # Get contact forces
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ff_contact = ff_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    mf_contact = mf_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    rf_contact = rf_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    lf_contact = lf_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    
    # Calculate contact magnitudes
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    ff_contact_mag = torch.norm(ff_contact, dim=-1)
    mf_contact_mag = torch.norm(mf_contact, dim=-1)
    rf_contact_mag = torch.norm(rf_contact, dim=-1)
    lf_contact_mag = torch.norm(lf_contact, dim=-1)


    thumb_in_contact = thumb_contact_mag > threshold
    ff_in_contact = ff_contact_mag > threshold
    mf_in_contact = mf_contact_mag > threshold
    rf_in_contact = rf_contact_mag > threshold
    lf_in_contact = lf_contact_mag > threshold


    if binary_contact:
        good_contact = thumb_in_contact & (
            ff_in_contact | 
            mf_in_contact | 
            rf_in_contact |
            lf_in_contact
        )
        return good_contact
    else:
        num_fingers_in_contact = (
            ff_in_contact.float() + 
            mf_in_contact.float() + 
            rf_in_contact.float() + 
            lf_in_contact.float()
        )
        normalized_finger_count = num_fingers_in_contact / 4.0
        contact_reward = thumb_in_contact.float() * normalized_finger_count
        return contact_reward


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()
