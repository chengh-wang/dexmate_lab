from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from .dexmate_lab_env_cfg import DexmateLabEnvCfg

@configclass
class ActionsIKCfg:
    """SE(3) IK control for teleoperation."""
    
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["R_arm_j.*"],    
        body_name="R_th_l0",          
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),  
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        scale=0.5,
    )
    # Binary gripper
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "R_th_j0", "R_th_j1",
            "R_ff_j1", "R_mf_j1",
            "R_rf_j1", "R_lf_j1",
        ],
        open_command_expr={"R_.*": 0.0},
        close_command_expr={"R_.*": 1.0},
    )


@configclass
class DexmateLabEnvCfgIK(DexmateLabEnvCfg):
    """IK variant for teleoperation."""
    
    actions: ActionsIKCfg = ActionsIKCfg()
    
    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 10.0