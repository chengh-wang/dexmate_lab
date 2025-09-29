# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import isaaclab.sim as sim_utils
from dataclasses import MISSING
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CuboidCfg, RigidBodyMaterialCfg
from isaaclab.utils import configclass  
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from . import mdp

##
# Pre-defined configs
##

from dexmate_lab.assets.robots.vega.vega_cfg import VEGA_CFG  # isort:skip
from dexmate_lab.assets.robots.vega.vega_cfg_right_arm import VEGA_RIGHT_ARM_CFG  # isort:skip

##
# Scene definition
##


@configclass
class DexmateLabSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # robot
    robot: ArticulationCfg = VEGA_RIGHT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.04),  # 6cm cube as required
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.5,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.29),  # Adjust based on robot reach
        ),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,  # Make visible for now
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Joint position control for right arm and hand
    joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "R_arm_j[1-7]",  # Right arm joints
            "R_th_j0",       # Thumb base
            "R_th_j1",       # Thumb flex
            "R_ff_j1",       # Fingers
            "R_mf_j1",
            "R_rf_j1",
            "R_lf_j1",
        ],
        scale=0.1,
        # use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 25

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 25

    @configclass
    class PerceptionObsCfg(ObsGroup):

        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 25

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # Table height randomization
    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.25, 0.25],  # Fixed x position
                "y": [0, 0.0],  # Small y variation
                "z": [0.05, 0.15],  # Height variation as required
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    # Cube placement randomization
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.25],   # Randomize on table surface
                "y": [-0.2, 0.15],
                "z": [0.15, 0.35],    # Small height above table (relative)
                "roll": [0, 0.],
                "pitch": [-0., 0.],
                "yaw": [0., 0.],
            },
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Reset robot joints with some variation
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["R_arm_j[1-7]"]),
            "position_range": [-10, 10],
            "velocity_range": [0.0, 0.0],
        },
    )

    # Physics randomization
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="R_.*"),
    #         "static_friction_range": [0.5, 1.0],
    #         "dynamic_friction_range": [0.5, 1.0],
    #         "restitution_range": [0.0, 0.0],
    #         "num_buckets": 250,
    #     },
    # )

    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "static_friction_range": [0.5, 1.0],
    #         "dynamic_friction_range": [0.5, 1.0],
    #         "restitution_range": [0.0, 0.0],
    #         "num_buckets": 250,
    #     },
    # )

    # variable_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="reset",
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    #         "operation": "abs",
    #     },
    # )

@configclass
class CurriculumCfg:

    adr = CurrTerm(
        func=mdp.DifficultyScheduler, params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )

    # gravity_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.variable_gravity.params.gravity_distribution_params",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
    #             "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )


@configclass
class CommandsCfg:
    """Command configuration."""
    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.2, 0.3),   # Reachable x range
            pos_y=(-0.1, 0.1),  # Reachable y range  
            pos_z=(0.4, 0.5),  # Above table height
            roll=(0.0, 0.0),    # No rotation for simple lifting
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Penalties
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    # Reaching reward
    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance, 
        params={
            "std": 0.4, 
            "asset_cfg": SceneEntityCfg("robot", body_names=[
                # "R_arm_l7",  # Wrist/end-effector
                "R_ff_l2", 
                "R_mf_l2",
                "R_rf_l2", 
                "R_lf_l2",
                "R_th_l2"
            ])
        }, 
        weight=2.0
    )

    # Grasping reward (based on contacts)
    good_grasp = RewTerm(
        func=mdp.contacts,
        weight=1.5,
        params={"threshold": 0.5, "binary_contact": False},
    )

    # Lifting reward
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=6.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Success reward
    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": None,  # No rotation for lifting task
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Termination penalty
    early_termination = RewTerm(
        func=mdp.is_terminated_term, 
        weight=-0.05, 
        params={"term_keys": ["object_out_of_bound"]}
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {
                "x": (-0.6, 1.0), 
                "y": (-0.7, 0.7),  
                "z": (0.0, 10.0)
            },
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

##
# Environment configuration
##


@configclass
class DexmateLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DexmateLabSceneCfg = DexmateLabSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        self.commands.object_pose.position_only = True
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15), roughness=0.25), visible=True
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15), roughness=0.25), visible=True
        )

        fingertip_links = [  # Change to list, not dictionary
            "R_th_l2",  # Thumb tip
            "R_ff_l2",  # Fore finger tip
            "R_mf_l2",  # Middle finger tip
            "R_rf_l2",  # Ring finger tip
            "R_lf_l2",  # Little finger tip
        ]

        for link_name in fingertip_links:
            setattr(
                self.scene,
                f"{link_name}_contact",  # Fixed: Use _object_s suffix consistently
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/.*{link_name}",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        
        # Add contact observations - now the names match
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_contact" for link in fingertip_links]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        
        # Update hand tips observation to include fingertips
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"] = SceneEntityCfg(
            "robot", 
            body_names=["R_arm_l7", "R_th_l2", "R_ff_l2", "R_mf_l2", "R_rf_l2", "R_lf_l2"]
        )