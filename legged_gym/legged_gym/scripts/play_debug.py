#!/usr/bin/env python3
"""
Debug play script with detailed logging for B1Z1 arm control
Outputs all joint actions, positions, velocities, torques, and EE tracking info
"""

import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.math import cart2sphere, sphere2cart

import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)  # Only 1 env for detailed logging
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.viewer.debug_viz = True  # Enable debug visualization
    
    # CRITICAL FIX: Set lenient termination thresholds to prevent early episode termination
    # This matches play.py settings so episodes can run longer for debugging
    env_cfg.termination.r_threshold = 1.0  # Allow roll up to 57 degrees (vs default 0.78)
    env_cfg.termination.p_threshold = 1.0  # Allow pitch up to 57 degrees (vs default 0.60)
    env_cfg.termination.z_threshold = 0.0  # Allow robot to touch ground (vs default 0.325)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print("\n" + "="*80)
    print("DEBUG MODE: Detailed B1Z1 Arm Control Logging")
    print("="*80)
    print(f"Z1 Stiffness: {env.p_gains[12:18]}")
    print(f"Z1 Damping: {env.d_gains[12:18]}")
    print(f"Action scale: {env.action_scale[12:18]}")
    # Handle both 1D and 2D default_dof_pos
    if env.default_dof_pos.dim() == 1:
        print(f"Default DOF pos (arm): {env.default_dof_pos[12:18]}")
    else:
        print(f"Default DOF pos (arm): {env.default_dof_pos[0, 12:18]}")
    print("="*80 + "\n")

    # Logging setup
    log_interval = 10  # Log every N steps
    step_counter = 0
    
    # Joint names for reference
    arm_joint_names = ['z1_waist', 'z1_shoulder', 'z1_elbow', 'z1_wrist_angle', 'z1_forearm_roll', 'z1_wrist_rotate']
    
    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        # Don't unpack - just call step and access env attributes directly
        env.step(actions.detach())
        obs = env.obs_buf
        
        # Detailed logging every log_interval steps
        if step_counter % log_interval == 0:
            env_id = 0  # Focus on first environment
            
            print(f"\n{'='*80}")
            print(f"STEP {step_counter}")
            print(f"{'='*80}")
            
            # 1. Policy Actions (what the network outputs)
            print(f"\n--- Policy Actions (raw, before scaling) ---")
            for j, name in enumerate(arm_joint_names):
                print(f"  {name:20s}: {actions[env_id, 12+j]:+.4f}")
            
            # 2. Scaled Actions (after action_scale)
            scaled_actions = actions[env_id, 12:18] * env.action_scale[12:18] * env.motor_strength[env_id, 12:18]
            print(f"\n--- Scaled Actions (action * scale * motor_strength) ---")
            for j, name in enumerate(arm_joint_names):
                print(f"  {name:20s}: {scaled_actions[j]:+.4f}")
            
            # 3. Current Joint States
            print(f"\n--- Current Joint States ---")
            # Handle both 1D and 2D default_dof_pos
            if env.default_dof_pos.dim() == 1:
                default_dof_pos_arm = env.default_dof_pos[12:18]
            else:
                default_dof_pos_arm = env.default_dof_pos[env_id, 12:18]
            
            for j, name in enumerate(arm_joint_names):
                pos = env.dof_pos[env_id, 12+j].item()
                vel = env.dof_vel[env_id, 12+j].item()
                default_pos = default_dof_pos_arm[j].item()
                print(f"  {name:20s}: pos={pos:+.4f} rad, vel={vel:+.4f} rad/s, default={default_pos:+.4f}")
            
            # 4. Target Positions (PD controller target)
            target_pos = scaled_actions + default_dof_pos_arm
            print(f"\n--- Target Joint Positions (scaled_action + default) ---")
            for j, name in enumerate(arm_joint_names):
                print(f"  {name:20s}: {target_pos[j]:+.4f} rad")
            
            # 5. Position Errors (for PD controller)
            pos_error = target_pos - env.dof_pos[env_id, 12:18]
            print(f"\n--- Position Errors (target - current) ---")
            for j, name in enumerate(arm_joint_names):
                print(f"  {name:20s}: {pos_error[j]:+.4f} rad")
            
            # 6. Applied Torques
            print(f"\n--- Applied Torques ---")
            for j, name in enumerate(arm_joint_names):
                torque = env.torques[env_id, 12+j].item()
                p_term = env.p_gains[12+j].item() * pos_error[j].item()
                d_term = -env.d_gains[12+j].item() * env.dof_vel[env_id, 12+j].item()
                print(f"  {name:20s}: {torque:+7.2f} Nm  (P:{p_term:+7.2f}, D:{d_term:+7.2f})")
            
            # 7. End-Effector Information
            print(f"\n--- End-Effector Tracking ---")
            
            # Current EE position in world frame (env already has ee_pos computed)
            ee_pos_world = env.ee_pos[env_id]
            print(f"  EE pos (world):        [{ee_pos_world[0]:+.4f}, {ee_pos_world[1]:+.4f}, {ee_pos_world[2]:+.4f}]")
            
            # Current EE position in local frame (body-relative)
            base_offset = torch.cat([env.root_states[env_id:env_id+1, :2], env.z_invariant_offset[env_id:env_id+1]], dim=1)
            ee_pos_local = quat_rotate_inverse(env.base_yaw_quat[env_id:env_id+1], 
                                               env.ee_pos[env_id:env_id+1] - base_offset)
            ee_sphere = cart2sphere(ee_pos_local)
            print(f"  EE pos (local cart):   [{ee_pos_local[0,0]:+.4f}, {ee_pos_local[0,1]:+.4f}, {ee_pos_local[0,2]:+.4f}]")
            print(f"  EE pos (sphere):       [r={ee_sphere[0,0]:+.4f}, pitch={ee_sphere[0,1]:+.4f}, yaw={ee_sphere[0,2]:+.4f}]")
            
            # Target EE position
            goal_sphere = env.curr_ee_goal_sphere[env_id]
            goal_cart = sphere2cart(goal_sphere.unsqueeze(0))
            print(f"  Goal (sphere):         [r={goal_sphere[0]:+.4f}, pitch={goal_sphere[1]:+.4f}, yaw={goal_sphere[2]:+.4f}]")
            print(f"  Goal (local cart):     [{goal_cart[0,0]:+.4f}, {goal_cart[0,1]:+.4f}, {goal_cart[0,2]:+.4f}]")
            
            # EE tracking error
            error_sphere = ee_sphere[0] - goal_sphere
            error_cart = ee_pos_local[0] - goal_cart[0]
            print(f"  Error (sphere):        [r={error_sphere[0]:+.4f}, pitch={error_sphere[1]:+.4f}, yaw={error_sphere[2]:+.4f}]")
            print(f"  Error (cart):          [{error_cart[0]:+.4f}, {error_cart[1]:+.4f}, {error_cart[2]:+.4f}]")
            print(f"  Error magnitude:       {torch.norm(error_cart):.4f} m")
            
            # 8. Interpretation
            print(f"\n--- Interpretation ---")
            if abs(goal_sphere[2]) > 0.1:
                direction = "LEFT" if goal_sphere[2] > 0 else "RIGHT"
                print(f"  Goal yaw = {goal_sphere[2]:+.4f} → Target is to the {direction}")
            else:
                print(f"  Goal yaw = {goal_sphere[2]:+.4f} → Target is FORWARD")
            
            if abs(actions[env_id, 12]) > 0.1:
                waist_direction = "LEFT" if actions[env_id, 12] > 0 else "RIGHT"
                print(f"  Waist action = {actions[env_id, 12]:+.4f} → Policy wants to move {waist_direction}")
            else:
                print(f"  Waist action = {actions[env_id, 12]:+.4f} → Policy wants to STAY")
            
            # Check if direction matches
            if abs(goal_sphere[2]) > 0.1 and abs(actions[env_id, 12]) > 0.1:
                goal_sign = 1 if goal_sphere[2] > 0 else -1
                action_sign = 1 if actions[env_id, 12] > 0 else -1
                if goal_sign == action_sign:
                    print(f"  ✅ Direction MATCHES: Policy correctly moving towards goal")
                else:
                    print(f"  ❌ Direction REVERSED: Policy moving OPPOSITE to goal!")
            
            # 9. Rewards
            print(f"\n--- Rewards ---")
            print(f"  Total reward:          {env.rew_buf[env_id].item():.4f}")
            if 'tracking_ee_sphere' in env.episode_sums:
                print(f"  Tracking EE (episode): {env.episode_sums['tracking_ee_sphere'][env_id].item():.4f}")
            
            print(f"{'='*80}\n")
        
        step_counter += 1

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
