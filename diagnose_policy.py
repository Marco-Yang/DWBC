#!/usr/bin/env python3
"""
Simple diagnostic: Load trained model and check if policy outputs large arm actions
"""

import sys
sys.path.append('legged_gym')
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry, get_args

def diagnose_policy(run_name, checkpoint):
    """Check what actions the trained policy outputs"""
    env_cfg, train_cfg = task_registry.get_cfgs(name='b1z1')
    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    
    # Lenient termination
    env_cfg.termination.r_threshold = 1.0
    env_cfg.termination.p_threshold = 1.0
    env_cfg.termination.z_threshold = 0.0
    
    # Create env
    env, _ = task_registry.make_env(name='b1z1', args=None, env_cfg=env_cfg)
    
    # Load policy
    train_cfg.runner.resume = True
    
    # Hack: set load_run and checkpoint
    class Args:
        def __init__(self):
            self.task = 'b1z1'
            self.load_run = run_name
            self.checkpoint = checkpoint
            
    args = Args()
    
    from legged_gym.utils.helpers import get_load_path
    load_path = get_load_path(root=f"legged_gym/logs/rough_{args.task}", 
                               load_run=args.load_run, 
                               checkpoint=args.checkpoint)
    
    print(f"Loading model from: {load_path}")
    
    from rsl_rl.runners import OnPolicyRunner
    ppo_runner = OnPolicyRunner(env, train_cfg.runner, log_dir=None, device='cuda:0')
    ppo_runner.load(load_path)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Run and collect statistics
    obs = env.reset()
    
    arm_actions = []
    ee_errors = []
    
    for i in range(200):
        with torch.no_grad():
            actions = policy(obs)
        
        # Record arm actions (indices 12-17, excluding gripper)
        arm_actions.append(actions[0, 12:18].cpu().numpy())
        
        # Step
        env.step(actions)
        obs = env.obs_buf
        
        # Calculate EE error
        ee_pos_local = env.ee_pos[0] - torch.cat([env.root_states[0, :2], env.z_invariant_offset[0:1]])
        ee_error = torch.norm(ee_pos_local[:2] - env.curr_ee_goal_cart[0, :2]).item()
        ee_errors.append(ee_error)
    
    # Statistics
    arm_actions = torch.tensor(arm_actions)
    
    print("\n" + "="*70)
    print("POLICY DIAGNOSIS:")
    print("="*70)
    
    print(f"\nArm Action Statistics (200 steps):")
    print(f"  Mean absolute: {arm_actions.abs().mean(dim=0)}")
    print(f"  Max absolute:  {arm_actions.abs().max(dim=0)[0]}")
    print(f"  Std dev:       {arm_actions.std(dim=0)}")
    
    # Check if actions are meaningful
    mean_abs = arm_actions.abs().mean()
    max_abs = arm_actions.abs().max()
    
    print(f"\nOverall:")
    print(f"  Mean |action|: {mean_abs:.4f}")
    print(f"  Max |action|:  {max_abs:.4f}")
    
    print(f"\nEE Tracking:")
    print(f"  Mean error: {torch.tensor(ee_errors).mean():.3f} m")
    print(f"  Min error:  {torch.tensor(ee_errors).min():.3f} m")
    print(f"  Final 50 steps mean: {torch.tensor(ee_errors[-50:]).mean():.3f} m")
    
    print("\n" + "="*70)
    print("VERDICT:")
    print("="*70)
    
    if mean_abs < 0.05:
        print("❌ Policy outputs VERY SMALL actions (<0.05)")
        print("   Policy hasn't learned to use the arm joints!")
        print("   Problem: Training not converged or reward signal too weak")
    elif mean_abs < 0.15:
        print("⚠️  Policy outputs small actions (0.05-0.15)")
        print("   Policy is cautious, may need more training")
    else:
        print("✅ Policy outputs reasonable actions (>0.15)")
        print("   Policy is actively using arm joints")
    
    if torch.tensor(ee_errors[-50:]).mean() < 0.15:
        print("\n✅ EE tracking is GOOD (<0.15m error)")
    elif torch.tensor(ee_errors[-50:]).mean() < 0.25:
        print("\n⚠️  EE tracking is OK (0.15-0.25m error)")
    else:
        print("\n❌ EE tracking is POOR (>0.25m error)")
    
    print("="*70)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_policy.py <run_name> [checkpoint]")
        print("Example: python diagnose_policy.py 8_b1z1_fd_fixied 1000")
        sys.exit(1)
    
    run_name = sys.argv[1]
    checkpoint = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    diagnose_policy(run_name, checkpoint)
