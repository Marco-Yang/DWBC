# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import torch

RESUME = False  # Set to True when resuming from a checkpoint

class B1Z1RoughCfg( LeggedRobotCfg ):
    """
    Configuration for B1Z1 robot (Unitree B1 + Unitree Z1 arm)
    - B1: 12 DOF legs (4 legs × 3 joints)
    - Z1: 7 DOF arm (6 arm joints + 1 gripper joint)
    - Total: 19 DOF (18 controllable actions, gripper not controlled)
    
    Key differences from WidowGo1:
    - Robot height: 0.5m (vs 0.42m for Go1)
    - Arm offset: z_invariant_offset = 0.7 (vs 0.53 for WidowX250s)
    - Gripper: 1 joint z1_jointGripper (vs 2 finger joints for WidowX250s)
    - Arm stiffness/damping tuned for Z1
    """

    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        # B1Z1 specific: Z1 arm workspace limits relative to sphere center
        collision_upper_limits = [0.1, 0.2, -0.05]  # Z1 arm workspace upper limits
        collision_lower_limits = [-0.8, -0.2, -0.7]  # Z1 arm workspace lower limits
        underground_limit = -0.7  # Z1 specific ground clearance
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.38  # Z1 specific: pitch induced by arm configuration

        class sphere_center:
            # B1Z1 specific: sphere center relative to robot base
            x_offset = 0.3  # Forward offset from base (meters)
            y_offset = 0  # Lateral offset from base (meters)
            z_invariant_offset = 0.7  # Z1 specific - height of sphere center (adjusted for B1 body height + Z1 mount)

        l_schedule = [0, 1]  # Curriculum learning schedule for radius
        p_schedule = [0, 1]  # Curriculum learning schedule for pitch
        y_schedule = [0, 1]  # Curriculum learning schedule for yaw
        tracking_ee_reward_schedule = [0, 1]  # Curriculum for tracking reward
        
        class ranges:
            # B1Z1 specific: Z1 arm reachable workspace in spherical coordinates
            # Initial EE positions (where trajectory starts)
            init_pos_start = [0.5, np.pi/8, 0]  # Initial EE position [radius, pitch, yaw]
            init_pos_end = [0.7, 0, 0]  # Initial EE position end range
            init_pos_l = [0.5, 0.7]  # Initial radius range for sampling
            init_pos_p = [0, np.pi/8]  # Initial pitch range for sampling
            init_pos_y = [0, 0]  # Initial yaw range (fixed at 0)
            
            # Final EE positions (where trajectory ends - full workspace)
            final_pos_l = [0.4, 0.95]  # Z1 reach range (radius in meters)
            final_pos_p = [-np.pi / 2.5, np.pi / 3]  # Z1 pitch range (radians)
            final_pos_y = [-1.2, 1.2]  # Z1 yaw range (radians)
            
            final_delta_orn = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]  # Z1 end-effector orientation deltas [roll, pitch, yaw]
            final_tracking_ee_reward = 0.55  # Target tracking reward weight

        # Error scaling for reward calculation
        sphere_error_scale = [
            1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 
            1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 
            1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])
        ]
        orn_error_scale = [2 / np.pi, 2 / np.pi, 2 / np.pi]
    
    class commands:
        curriculum = True
        num_commands = 3  # [lin_vel_x, lin_vel_y (unused), ang_vel_yaw]
        resampling_time = 3.  # Time before commands are changed [s]

        # Curriculum learning schedules
        lin_vel_x_schedule = [0, 1]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        # Command clipping for safety
        ang_vel_yaw_clip = 0.6
        lin_vel_x_clip = 0.3

        class ranges:
            # Command ranges for curriculum learning
            final_lin_vel_x = [0, 0.9]  # Final linear velocity X range [m/s]
            final_ang_vel_yaw = [-1.0, 1.0]  # Final angular velocity yaw range [rad/s]
            init_lin_vel_x = [0, 0]  # Initial linear velocity (start from stationary)
            init_ang_vel_yaw = [0, 0]  # Initial angular velocity (start from stationary)
            
            final_tracking_ang_vel_yaw_exp = 0.15  # Target tracking reward for angular velocity

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 5000
        num_actions = 12 + 6  # B1Z1: 12 leg DOF + 6 arm DOF (gripper not controlled)
        num_torques = 12 + 6  # Same as num_actions
        action_delay = 2  # Number of steps to delay actions (-1 for no delay)
        
        # B1Z1: 19 total DOF (12 legs + 7 arm including gripper)
        # Observation breakdown:
        # - 2: body orientation (roll, pitch)
        # - 3: angular velocity
        # - 19: DOF positions
        # - 19: DOF velocities
        # - 18: action history (previous action, no gripper)
        # - 4: foot contacts
        # - 3: locomotion commands [lin_vel_x, lin_vel_y, ang_vel_yaw]
        # - 3: CURRENT EE position in sphere coordinates [radius, pitch, yaw]
        # - 3: TARGET EE goal in sphere coordinates [radius, pitch, yaw]
        # - 3: EE orientation deltas [roll, pitch, yaw]
        # Total proprio: 2 + 3 + 19 + 19 + 18 + 4 + 3 + 3 + 3 + 3 = 77
        num_proprio = 2 + 3 + 19 + 19 + 18 + 4 + 3 + 3 + 3 + 3
        num_priv = 5 + 1 + 18  # Domain randomization: [mass(5), friction(1), motor_strength(18)]
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv  # = 77*11 + 24 = 871

        num_privileged_obs = None  # None = no privileged observations
        send_timeouts = True
        episode_length_s = 10  # Episode length in seconds

        reorder_dofs = True  # Reorder DOFs to match A1 convention (FR, FL, RR, RL)

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m] - B1 is taller than Go1 (0.42m)
        default_joint_angles = {  # Target angles [rad] when action = 0.0
            # B1 leg joints (same naming convention as Go1)
            'FL_hip_joint': 0.2,
            'FL_thigh_joint': 0.8,
            'FL_calf_joint': -1.5,

            'RL_hip_joint': 0.2,
            'RL_thigh_joint': 0.8,
            'RL_calf_joint': -1.5,

            'FR_hip_joint': -0.2,
            'FR_thigh_joint': 0.8,
            'FR_calf_joint': -1.5,

            'RR_hip_joint': -0.2,
            'RR_thigh_joint': 0.8,
            'RR_calf_joint': -1.5,

            # Z1 arm joints (6 DOF + 1 gripper)
            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -0.63,
            'z1_wrist_angle': -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57,
            'z1_jointGripper': -0.785,  # Z1 has only 1 gripper joint (vs 2 for WidowX250s)
        }

    class control:
        # PD Drive parameters tuned for B1Z1
        stiffness = {'joint': 50, 'z1': 30}  # [N*m/rad] - Z1 uses higher stiffness for better tracking
        damping = {'joint': 1, 'z1': 2}  # [N*m/s/rad] - Z1 damping increased for stability
        adaptive_arm_gains = False  # Whether to learn arm gains
        
        # Action scale: target angle = action_scale * action + default_angle
        # B1Z1 specific: Legs use 0.4-0.45, Z1 arm joints need MUCH larger scales for fast movement
        # CRITICAL FIX v2: Further increased Z1 scales from 1.5 to 2.5 for more aggressive arm control
        # Reasoning: With stiffness=30, we need larger action range to enable fast EE tracking
        # Policy output ∈ [-1,1], so actual range = ±2.5 rad = ±143 degrees (plenty of range)
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 2.5, 2.5, 2.5, 2.5, 2.5]
        # Breakdown: [FL_hip, FL_thigh, FL_calf] + [FR_hip, FR_thigh, FR_calf] + 
        #            [RL_hip, RL_thigh, RL_calf] + [RR_hip, RR_thigh, RR_calf] +
        #            [z1_waist(2.1), z1_shoulder(2.5), z1_elbow(2.5), z1_wrist_angle(2.5), z1_forearm_roll(2.5), z1_wrist_rotate(2.5)]
        # Note: Gripper joint is controlled separately, not included in actions
        
        decimation = 4  # Number of control action updates @ sim DT per policy DT
        torque_supervision = False  # Whether to use torque supervision

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b1z1.urdf'  # B1Z1 URDF file
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "trunk"]  # Body parts to penalize contact
        terminate_after_contacts_on = []  # No termination on contact
        self_collisions = 0  # 0 = enable self-collisions, 1 = disable
        flip_visual_attachments = False
        collapse_fixed_joints = True
        fix_base_link = False
    
    class box:
        # Object manipulation settings (currently not used)
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
        box_pos_obs_range = 1.0
    
    class arm:
        # Arm control parameters
        init_target_ee_base = [0.2, 0.0, 0.2]  # Initial target EE position relative to base
        grasp_offset = 0.08  # Grasp offset for object manipulation
        osc_kp = np.array([100, 100, 100, 30, 30, 30])  # OSC position/orientation gains
        osc_kd = 2 * (osc_kp ** 0.5)  # OSC damping (critically damped)
        
        # B1Z1 specific: Z1 arm base offset relative to robot trunk (currently unused but kept for future use)
        arm_base_overhead = [0., 0., 0.165]  # [x, y, z] offset of arm base from trunk

    class domain_rand:
        observe_priv = True  # Whether to observe privileged information (for asymmetric training)
        
        # Friction randomization
        randomize_friction = True
        friction_range = [-0.5, 3.0]
        
        # Base mass randomization
        randomize_base_mass = True
        added_mass_range = [-0.5, 2.5]
        
        # Center of mass randomization
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        
        # Motor strength randomization
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]

        # Gripper mass randomization (for payload simulation)
        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]

        # External force perturbations
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 0.5

        cube_y_range = [0.2, 0.4]
        
    class noise( LeggedRobotCfg.noise ):
        add_noise = False  # Whether to add noise to observations
  
    class rewards:
        class scales:
            # Locomotion rewards
            termination = -0
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = -0.
            ang_vel_xy = -0.
            orientation = -0.
            torques = 0
            energy_square = -6e-5
            dof_vel = 0
            dof_acc = -0
            base_height = 0
            feet_air_time = 0
            collision = 0
            feet_stumble = -0 
            action_rate = -0
            stand_still = 0
            survive = 0.2
            leg_energy = -0
            leg_energy_abs_sum = -0
            tracking_lin_vel_x_l1 = 0.5
            tracking_lin_vel_x_exp = 0.
            tracking_ang_vel_yaw_l1 = 0
            tracking_ang_vel_yaw_exp = 0.15
            tracking_lin_vel_y_l2 = 0
            tracking_lin_vel_z_l2 = -0.0
            leg_action_l2 = -0.0
            hip_action_l2 = -0.01
            foot_contacts_z = -1e-4
            
        class arm_scales:
            # Arm manipulation rewards
            termination = -0.0
            tracking_ee_sphere = 0.55  # Main EE tracking reward in sphere coordinates
            tracking_ee_cart = 0.0  # Cartesian tracking reward (not used)
            arm_orientation = -0.
            arm_energy_abs_sum = -0.0040  # Energy penalty for arm
            tracking_ee_orn = 0.  # Orientation tracking reward
            tracking_ee_orn_ry = 0.  # Roll-yaw orientation tracking
        
        only_positive_rewards = False
        tracking_sigma = 1  # Gaussian kernel width for tracking rewards
        tracking_ee_sigma = 1  # Gaussian kernel width for EE tracking rewards
        soft_dof_pos_limit = 1.  # Percentage of URDF limits to start penalizing
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25  # Target base height for base_height reward
        max_contact_force = 100.  # Forces above this value are penalized

    class viewer:
        pos = [-20, 0, 20]  # [m] - Camera position
        lookat = [0, 0, -2]  # [m] - Camera look-at point

    class termination:
        # B1Z1 specific termination thresholds (adjusted for B1 height)
        r_threshold = 0.78  # Roll termination threshold [rad]
        p_threshold = 0.60  # Pitch termination threshold [rad]
        z_threshold = 0.325  # Height termination threshold [m]

    class terrain:
        mesh_type = 'plane'  # Options: 'plane', 'heightfield', 'trimesh'
        add_slopes = True
        slope_incline = 0.2
        horizontal_scale = 0.025  # [m]
        vertical_scale = 1 / 100000  # [m]
        border_size = 0  # [m]
        tot_cols = 600
        tot_rows = 10000
        zScale = 0.15
        transform_x = - tot_cols * horizontal_scale / 2
        transform_y = - tot_rows * horizontal_scale / 2
        transform_z = 0.0

        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
        # Rough terrain settings (not used with plane terrain)
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        slope_treshold = 100000000  # Slopes above this threshold become vertical

        # Initial state perturbations
        origin_perturb_range = 0.5  # [m]
        init_vel_perturb_range = 0.1  # [m/s]


class B1Z1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        # Initial action standard deviation for exploration
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]  # Legs + arm
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu'  # Options: elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # Separate control heads for legs and arm
        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        # Privileged information encoder (for asymmetric training)
        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0

    class algorithm:
        # PPO algorithm parameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2e-4
        schedule = 'fixed'  # Options: 'adaptive', 'fixed'
        gamma = 0.99  # Discount factor
        lam = 0.95  # GAE lambda
        desired_kl = None
        max_grad_norm = 1.
        
        # Minimum policy standard deviation (for stability)
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        # Curriculum learning schedules
        mixing_schedule = [1.0, 0, 3000] if not RESUME else [1.0, 0, 1]
        torque_supervision = B1Z1RoughCfg.control.torque_supervision
        torque_supervision_schedule = [0.0, 1000, 1000]
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains

        # DAgger (privileged learning) parameters
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] if not RESUME else [0, 1, 1000, 1000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 40  # Number of steps to collect per environment per update
        max_iterations = 40000  # Maximum number of policy updates

        # Logging and checkpointing
        save_interval = 500  # Save model every N iterations
        experiment_name = 'rough_b1z1'
        run_name = ''
        
        # Loading and resuming
        resume = RESUME
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # Will be set based on load_run and checkpoint
