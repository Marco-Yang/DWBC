# B1Z1机械臂控制问题诊断与修复总结

## 发现的关键问题

### 🔴 CRITICAL BUG #1: 观测缺少当前EE位置
**问题**：
- 策略网络观测中只有目标EE位置（`curr_ee_goal_sphere`），没有当前EE位置
- 策略无法知道"现在在哪里"，只知道"要去哪里"
- 这使得学习EE追踪几乎不可能

**修复**：
```python
# 在compute_observations()中添加：
ee_pos_local = quat_rotate_inverse(self.base_yaw_quat, 
                                   self.ee_pos - torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1))
curr_ee_sphere = cart2sphere(ee_pos_local)  # [r, pitch, yaw]

obs_buf = torch.cat((
    ...
    curr_ee_sphere,       # dim 3 - CURRENT EE position (NEW!)
    self.curr_ee_goal,    # dim 3 - TARGET EE position
    ...
))
```

**影响**：
- 观测维度：74 → 77
- num_observations：838 → 871
- **需要重新训练所有模型**

---

### 🔴 CRITICAL BUG #2: Z1 PD gains太低
**问题**：
- Z1 stiffness = 5 [N*m/rad]
- Z1 damping = 0.5 [N*m*s/rad]  
- 这些值导致机械臂产生的力矩太小，无法快速移动

**对比**：
- 腿部：stiffness=50, damping=1
- Z1手臂比WidowX250s更重，需要更高的gains

**修复**：
```python
class control:
    stiffness = {'joint': 50, 'z1': 30}  # 5 → 30 (6x increase)
    damping = {'joint': 1, 'z1': 2}      # 0.5 → 2 (4x increase)
```

**预期效果**：
- 机械臂响应速度应该提升~5-6倍
- 能够在50步内移动>0.5 rad
- 更稳定的控制（更高damping减少震荡）

---

## 训练历史

### ❌ 运行5 (6_b1z1_fd_fixied, 1000 iterations)
**状态**: 失败
**问题**:
1. ✅ URDF fixes已应用
2. ❌ 观测缺少当前EE位置 → 无法学习
3. ❌ PD gains太低 → 手臂无力

**结果**:
- Train/mean_arm_reward: 0 → 0.022 (几乎没有进步)
- tracking_ee_sphere: 维持在0.22-0.25 (没有改善)
- 机械臂几乎不动

---

## 待验证的修复

### ✅ 修复清单
1. [x] URDF dynamics: damping=0, friction=0
2. [x] action_scale: [2.1, 0.6, 0.6, 0.6, 0.6, 0.6]
3. [x] arm_base_idx = 18
4. [x] DOF indexing: -6 for Z1 arm
5. [x] 观测添加curr_ee_sphere
6. [x] num_observations: 838 → 871
7. [x] Z1 stiffness: 5 → 30
8. [x] Z1 damping: 0.5 → 2

### 📋 下一步行动

#### 1. 验证新PD gains
```bash
python test_arm_gains.py
```
**预期结果**: 
- Waist joint在50步内移动>0.5 rad
- 所有6个arm joints都能响应动作

#### 2. 启动新训练 (运行7)
```bash
export PYTORCH_JIT=0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python legged_gym/scripts/train.py \
    --task=b1z1 \
    --exptid=7 \
    --run_name=b1z1_full_fix \
    --headless
```

#### 3. 监控指标
**关键指标** (应该在前500次迭代内看到明显改善):
- `Train/mean_arm_reward`: 应该从0快速涨到>0.3
- `Episode/rew_tracking_ee_sphere`: 应该从~0.25下降到<0.15
- 所有6个arm joints应该有显著的position variance

**对比基准** (之前失败的训练):
- mean_arm_reward最高只到0.022
- tracking_ee_sphere停滞在0.22-0.25
- 只有waist在动，其他关节几乎静止

#### 4. 测试检查点
在500次迭代后测试：
```bash
python legged_gym/scripts/play.py \
    --task=b1z1 \
    --load_run=7_b1z1_full_fix \
    --checkpoint=500
```

**验证要点**:
- [ ] 机械臂能快速移动到目标位置（红色球）
- [ ] 所有6个arm joints都在活动
- [ ] Waist方向正确（目标在左就向左转）
- [ ] 末端执行器能到达工作空间的不同位置

---

## 技术细节

### PD控制器公式
```python
torque = stiffness * (target_pos - current_pos) - damping * current_vel
```

### Z1手臂工作空间 (球坐标)
- 半径r: [0.4, 0.95] m
- 俯仰pitch: [-π/2.5, π/3] rad
- 偏航yaw: [-1.2, 1.2] rad

### 坐标系统
- yaw > 0 → 左侧 (+y方向)
- yaw < 0 → 右侧 (-y方向)  
- pitch > 0 → 向上
- pitch < 0 → 向下

### 观测维度详解 (77维)
```
2: body orientation (roll, pitch)
3: angular velocity
19: DOF positions
19: DOF velocities
18: action history
4: foot contacts
3: locomotion commands
3: CURRENT EE position (sphere) ← NEW!
3: TARGET EE goal (sphere)
3: EE orientation deltas
---
77 total
```

---

## 预期改进

### 训练速度
- 之前：1000次迭代几乎没有arm学习
- 预期：500次迭代应该看到明显的arm控制

### 最终性能
- Arm reward应该达到0.5-0.7 (vs 0.022)
- EE tracking error应该<0.1m (vs >0.2m)
- 所有6个arm DOF协同工作完成复杂操作

---

## 故障排查

如果新训练还是失败，检查：
1. [ ] PD gains是否生效: `print(env.p_gains[12:18])`
2. [ ] 观测维度是否正确: `print(obs.shape)`  
3. [ ] 机械臂是否能产生足够力矩: 运行test_arm_gains.py
4. [ ] URDF dynamics是否正确: damping=0, friction=0

---

**日期**: 2024-11-27  
**状态**: 准备开始运行7训练
