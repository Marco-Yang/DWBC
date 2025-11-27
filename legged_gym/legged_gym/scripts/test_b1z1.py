#!/usr/bin/env python

"""
Test script to verify B1Z1 environment configuration
"""

import isaacgym
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def test_b1z1_env():
    """Test B1Z1 environment creation"""
    print("\n" + "="*50)
    print("Testing B1Z1 Environment Configuration")
    print("="*50 + "\n")
    
    # Get configuration
    env_cfg, train_cfg = task_registry.get_cfgs(name='b1z1')
    
    print("✓ Successfully loaded B1Z1 configuration")
    print(f"  - URDF file: {env_cfg.asset.file}")
    print(f"  - Num actions: {env_cfg.env.num_actions}")
    print(f"  - Num torques: {env_cfg.env.num_torques}")
    print(f"  - Init position: {env_cfg.init_state.pos}")
    
    # Override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    
    print("\n" + "-"*50)
    print("Creating Environment...")
    print("-"*50 + "\n")
    
    # Create environment
    env,_ = task_registry.make_env(name='b1z1', args=None, env_cfg=env_cfg)
    
    print("✓ Successfully created B1Z1 environment!")
    print(f"  - Number of DOFs: {env.num_dofs}")
    print(f"  - DOF names: {env.dof_names}")
    print(f"  - Number of bodies: {env.num_bodies}")
    print(f"  - Foot indices: {env.feet_indices}")
    print(f"  - Gripper index: {env.gripper_idx if hasattr(env, 'gripper_idx') else 'N/A'}")
    
    print("\n" + "-"*50)
    print("Testing Environment Step...")
    print("-"*50 + "\n")
    
    # Test a few steps
    obs, privileged_obs = env.reset()
    if privileged_obs is not None:
        print(f"✓ Reset successful, obs shape: {obs.shape}, privileged_obs shape: {privileged_obs.shape}")
    else:
        print(f"✓ Reset successful, obs shape: {obs.shape}, privileged_obs: None")
    
    for i in range(5):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, privileged_obs, rewards, arm_rewards, dones, infos = env.step(actions)
        print(f"  Step {i+1}: reward={rewards[0].item():.4f}, arm_reward={arm_rewards[0].item():.4f}, done={dones[0].item()}")
    
    print("\n" + "="*50)
    print("B1Z1 Environment Test PASSED! ✓")
    print("="*50 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = test_b1z1_env()
        if success:
            print("\n✅ All tests passed successfully!")
        else:
            print("\n❌ Some tests failed")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
