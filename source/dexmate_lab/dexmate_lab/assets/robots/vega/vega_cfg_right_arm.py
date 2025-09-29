# source/dexmate_lab/dexmate_lab/assets/robots/vega/vega_cfg_right_arm.py

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

VEGA_RIGHT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path="source/dexmate_lab/dexmate_lab/assets/robots/vega/vega_upper_body_v1.usd",
        usd_path="/home/chengh-wang/Documents/git/dexmate_lab/source/dexmate_lab/dexmate_lab/assets/robots/vega/vega_upper_body_right_arm_v1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Initialize right arm in pre-grasp position
            # "R_arm_j1": 0.523,
            # "R_arm_j2": -0.349,  # Slightly raised
            # "R_arm_j3": -1.57,
            # "R_arm_j4": -1.04,  # Elbow bent
            # "R_arm_j5": 0.0,
            # "R_arm_j6": 0.0,
            # "R_arm_j7": 0.0,
            "R_arm_j1": 0.,
            "R_arm_j2": -0.,  # Slightly raised
            "R_arm_j3": -0.,
            "R_arm_j4": -0.,  # Elbow bent
            "R_arm_j5": 0.0,
            "R_arm_j6": 0.0,
            "R_arm_j7": 0.0,
            
            # Right hand fingers open (only j1 joints, j2 will follow via mimic)
            "R_th_j0": 0.,  # Thumb base
            "R_th_j1": 0.183,  # Thumb finger
            "R_ff_j1": 0.279,  # Fore finger
            "R_mf_j1": 0.279,  # Middle finger
            "R_rf_j1": 0.279,  # Ring finger
            "R_lf_j1": 0.279,  # Little finger
            
            # Head in neutral
            # "head_j1": 0.0,
            # "head_j2": 0.0,
            # "head_j3": 0.0,
        },
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["R_arm_j[1-7]"],
            effort_limit={
                "R_arm_j[1-2]": 150.0,  # Shoulder joints
                "R_arm_j[3-4]": 80.0,   # Elbow joints
                "R_arm_j[5-7]": 25.0,   # Wrist joints
            },
            stiffness={
                "R_arm_j[1-2]": 500.0,
                "R_arm_j[3-4]": 500.0,
                "R_arm_j[5-7]": 500.0,
            },
            damping={
                "R_arm_j[1-2]": 40.0,
                "R_arm_j[3-4]": 30.0,
                "R_arm_j[5-7]": 10.0,
            },
        ),
        "right_hand": ImplicitActuatorCfg(
            # Only actuate j1 and j0 joints - j2 joints will follow via mimic
            joint_names_expr=[
                "R_th_j0",  # Thumb base rotation
                "R_th_j1",  # Thumb flex (j2 will mimic)
                "R_ff_j1",  # Fingers only need j1 (j2 will mimic)
                "R_mf_j1",
                "R_rf_j1", 
                "R_lf_j1",
            ],
            effort_limit=1.0,  # From URDF
            stiffness=50.0,
            damping=0.5,
            friction=0.01,
        ),
        # Don't actuate left side - just leave it unactuated
        # The simulator will handle unactuated joints based on physics
    },
    soft_joint_pos_limit_factor=1.0,
)