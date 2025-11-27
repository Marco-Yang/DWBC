#!/usr/bin/env python3
"""
Test script to verify action_scale=2.5 enables faster arm movement than 1.5
Tests different scales in a SINGLE environment to avoid IsaacGym conflicts
"""

import sys
sys.path.append('legged_gym')
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import task_registry

import torch

def test_multiple_action_scales(scales_to_test, num_steps=50):
    """Test arm movement with different action scales in ONE environment"""
    # Get env config - use the largest scale to create env
    env_cfg, _ = task_registry.get_cfgs(name='b1z1')
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Set lenient termination to prevent early resets
    env_cfg.termination.r_threshold = 1.0
    env_cfg.termination.p_threshold = 1.0
    env_cfg.termination.z_threshold = 0.0
    
    # Use max scale for env creation (we'll adjust manually)
    max_scale = max(scales_to_test)
    original_scale = env_cfg.control.action_scale
    env_cfg.control.action_scale = original_scale[:12] + [2.1] + [max_scale]*5
    
    # Create env ONCE
    env, _ = task_registry.make_env(name='b1z1', args=None, env_cfg=env_cfg)
    
    results = {}
    
    for action_scale_value in scales_to_test:
        print(f"\nTesting action_scale = {action_scale_value}...")
        
        # Reset environment
        env.reset()
        
        # Record initial arm positions
        initial_pos = env.dof_pos[0, 12:18].clone()
        
        # Apply constant action to all arm joints (except waist)
        # Scale the action to simulate different action_scale values
        action = torch.zeros(1, 19, device=env.device)
        # Manually scale: effective_action = policy_output * (test_scale / max_scale)
        scale_ratio = action_scale_value / max_scale
        action[0, 13:18] = 0.5 * scale_ratio  # Adjust action to simulate smaller scale
        
        # Run simulation
        for _ in range(num_steps):
            env.step(action)
        
        # Measure final arm positions
        final_pos = env.dof_pos[0, 12:18]
        
        # Calculate movement
        movement = torch.abs(final_pos - initial_pos)
        total_movement = movement.sum().item()
        max_movement = movement.max().item()
        
        results[action_scale_value] = {
            'total_movement': total_movement,
            'max_movement': max_movement,
            'per_joint_movement': movement.cpu().numpy()
        }
        
        print(f"  Total movement: {total_movement:.3f} rad")
        print(f"  Max single joint: {max_movement:.3f} rad")
    
    return results

if __name__ == '__main__':
    print("="*70)
    print("Testing Different Action Scales for Z1 Arm")
    print("="*70)
    print(f"Test: Apply action=0.5 to all arm joints for 50 steps")
    print(f"Goal: Verify larger action_scale enables faster movement")
    print()
    
    scales_to_test = [0.6, 1.5, 2.5]
    
    print(f"Testing scales: {scales_to_test}")
    results = test_multiple_action_scales(scales_to_test, num_steps=50)
    
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)
    
    for scale in scales_to_test:
        r = results[scale]
        print(f"\naction_scale = {scale}:")
        print(f"  Total: {r['total_movement']:.3f} rad")
        print(f"  Max: {r['max_movement']:.3f} rad")
        
        # Compare to baseline (0.6)
        if scale != 0.6:
            improvement = r['total_movement'] / results[0.6]['total_movement']
            print(f"  Improvement vs 0.6: {improvement:.1f}x")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    if results[2.5]['total_movement'] > results[1.5]['total_movement'] * 1.3:
        print("✅ action_scale=2.5 provides significantly more movement!")
        print("   Recommend using 2.5 for training.")
    elif results[2.5]['total_movement'] > results[1.5]['total_movement']:
        print("✅ action_scale=2.5 provides more movement than 1.5")
        print("   Improvement is modest but helpful.")
    else:
        print("⚠️  action_scale=2.5 doesn't provide much benefit over 1.5")
        print("   May have hit other limits (PD gains, joint limits, etc.)")
    print("="*70)
