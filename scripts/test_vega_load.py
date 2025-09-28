# scripts/test_env_viz.py

import torch
from isaaclab.app import AppLauncher

# Add command line args
import argparse
parser = argparse.ArgumentParser(description="Test Vega environment visualization.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Dexmate-Lab-v0", help="Name of the task.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import after app is created
import gymnasium as gym
import dexmate_lab.tasks.manager_based.dexmate_lab  # This triggers registration
from isaaclab_tasks.utils import parse_env_cfg

def main():
    """Test the environment with zero actions."""
    
    # Parse environment configuration properly
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=True
    )
    
    # Create environment with parsed config
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    
    # Reset environment
    obs, info = env.reset()
    print("Environment reset successful!")
    
    # Run with zero actions for visualization
    for i in range(1000):
        # Zero actions to just observe the scene
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if i % 100 == 0:
            print(f"Step {i}: Reward mean = {reward.mean():.3f}")
        
        # The environment handles resets automatically in vectorized case
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()