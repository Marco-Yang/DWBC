# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

# @ torch.jit.script
def torch_rand_sign(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    """Generate random signs (+1 or -1) with given shape"""
    return torch.where(torch.rand(*shape, device=device) > 0.5, 
                      torch.ones(*shape, device=device), 
                      -torch.ones(*shape, device=device))

# @ torch.jit.script
def euler_from_quat(quat):
    """
    Convert quaternion to euler angles (roll, pitch, yaw)
    quat: (N, 4) tensor in [x, y, z, w] format
    returns: tuple of (roll, pitch, yaw) tensors
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1,
                       torch.sign(sinp) * np.pi / 2,
                       torch.asin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

# @ torch.jit.script  
def quat_from_euler_xyz(roll, pitch, yaw):
    """
    Convert euler angles to quaternion
    returns: (N, 4) tensor in [x, y, z, w] format
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return torch.stack([x, y, z, w], dim=-1)

# @ torch.jit.script
def torch_wrap_to_pi_minuspi(angles):
    """Wrap angles to [-pi, pi]"""
    angles = angles % (2 * np.pi)
    angles = torch.where(angles > np.pi, angles - 2 * np.pi, angles)
    return angles

# @ torch.jit.script
def cart2sphere(cart_coords):
    """
    Convert Cartesian coordinates to spherical coordinates
    cart_coords: (..., 3) tensor with [x, y, z]
    returns: (..., 3) tensor with [r, theta, phi]
        r: radial distance
        theta: polar angle (angle from z-axis), range [0, pi]
        phi: azimuthal angle (angle in xy-plane from x-axis), range [-pi, pi]
    """
    x = cart_coords[..., 0]
    y = cart_coords[..., 1]
    z = cart_coords[..., 2]
    
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))  # polar angle
    phi = torch.atan2(y, x)  # azimuthal angle
    
    return torch.stack([r, theta, phi], dim=-1)

# @ torch.jit.script
def sphere2cart(sphere_coords):
    """
    Convert spherical coordinates to Cartesian coordinates
    sphere_coords: (..., 3) tensor with [r, theta, phi]
        r: radial distance
        theta: polar angle (angle from z-axis)
        phi: azimuthal angle (angle in xy-plane from x-axis)
    returns: (..., 3) tensor with [x, y, z]
    """
    r = sphere_coords[..., 0]
    theta = sphere_coords[..., 1]
    phi = sphere_coords[..., 2]
    
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    
    return torch.stack([x, y, z], dim=-1)