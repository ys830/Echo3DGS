import torch
import math
from us_gaussian_voxelization import (
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
from scene.gaussian_model import GaussianModel
from torch import nn

# def quaternion_multiplys(q, r):
#     """
#     Multiply two sets of quaternions q and r.
#     q: (90, 1000, 4) 四元数张量
#     r: (90, 1000, 4) 四元数张量
#     返回:
#     结果: (90, 1000, 4) 四元数张量，表示每对 (q, r) 的乘积
#     """
#     # 提取每个四元数的 w, x, y, z 分量
#     w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
#     w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    
#     # 计算四元数乘法结果
#     result = torch.stack((
#         w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
#         w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
#         w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
#         w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
#     ), dim=-1)
    
#     return result



# def rotation_matrices_to_quaternions(r_matrices):
#     N = r_matrices.shape[0]
#     quaternions = torch.zeros((N, 4), device=r_matrices.device)

#     trace = r_matrices[:, 0, 0] + r_matrices[:, 1, 1] + r_matrices[:, 2, 2]
#     mask_trace_positive = trace > 0

#     # 对trace > 0的情况进行处理
#     s = torch.sqrt(trace[mask_trace_positive] + 1.0+ 1e-6) * 0.5
#     quaternions[mask_trace_positive, 0] = s
#     quaternions[mask_trace_positive, 1] = (r_matrices[mask_trace_positive, 2, 1] - r_matrices[mask_trace_positive, 1, 2]) / (4.0 * s)
#     quaternions[mask_trace_positive, 2] = (r_matrices[mask_trace_positive, 0, 2] - r_matrices[mask_trace_positive, 2, 0]) / (4.0 * s)
#     quaternions[mask_trace_positive, 3] = (r_matrices[mask_trace_positive, 1, 0] - r_matrices[mask_trace_positive, 0, 1]) / (4.0 * s)

#     # 对trace <= 0的情况进行处理
#     mask_col0 = (r_matrices[:, 0, 0] > r_matrices[:, 1, 1]) & (r_matrices[:, 0, 0] > r_matrices[:, 2, 2])
#     mask_col1 = (r_matrices[:, 1, 1] > r_matrices[:, 2, 2]) & ~mask_col0
#     mask_col2 = ~(mask_col0 | mask_col1)

#     # 处理 r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2] 的情况
#     s_col0 = torch.sqrt(1.0 + r_matrices[mask_col0, 0, 0] - r_matrices[mask_col0, 1, 1] - r_matrices[mask_col0, 2, 2]) * 2.0
#     quaternions[mask_col0, 0] = (r_matrices[mask_col0, 2, 1] - r_matrices[mask_col0, 1, 2]) / s_col0
#     quaternions[mask_col0, 1] = 0.25 * s_col0
#     quaternions[mask_col0, 2] = (r_matrices[mask_col0, 0, 1] + r_matrices[mask_col0, 1, 0]) / s_col0
#     quaternions[mask_col0, 3] = (r_matrices[mask_col0, 0, 2] + r_matrices[mask_col0, 2, 0]) / s_col0

#     # 处理 r[1, 1] > r[2, 2] 的情况
#     s_col1 = torch.sqrt(1.0 + r_matrices[mask_col1, 1, 1] - r_matrices[mask_col1, 0, 0] - r_matrices[mask_col1, 2, 2]) * 2.0
#     quaternions[mask_col1, 0] = (r_matrices[mask_col1, 0, 2] - r_matrices[mask_col1, 2, 0]) / s_col1
#     quaternions[mask_col1, 1] = (r_matrices[mask_col1, 0, 1] + r_matrices[mask_col1, 1, 0]) / s_col1
#     quaternions[mask_col1, 2] = 0.25 * s_col1
#     quaternions[mask_col1, 3] = (r_matrices[mask_col1, 1, 2] + r_matrices[mask_col1, 2, 1]) / s_col1

#     # 处理剩下的情况 (r[2, 2] 最大的情况)
#     s_col2 = torch.sqrt(1.0 + r_matrices[mask_col2, 2, 2] - r_matrices[mask_col2, 0, 0] - r_matrices[mask_col2, 1, 1]) * 2.0
#     quaternions[mask_col2, 0] = (r_matrices[mask_col2, 1, 0] - r_matrices[mask_col2, 0, 1]) / s_col2
#     quaternions[mask_col2, 1] = (r_matrices[mask_col2, 0, 2] + r_matrices[mask_col2, 2, 0]) / s_col2
#     quaternions[mask_col2, 2] = (r_matrices[mask_col2, 1, 2] + r_matrices[mask_col2, 2, 1]) / s_col2
#     quaternions[mask_col2, 3] = 0.25 * s_col2

#     # 归一化四元数
#     norms = quaternions.norm(dim=1, keepdim=True) + 1e-6
#     quaternions = quaternions / norms
    
#     return quaternions



# def query(pc: GaussianModel, center, nVoxel, sVoxel,Poses, pipe, scaling_modifier=1.0):
#     voxel_settings = GaussianVoxelizationSettings(
#         scale_modifier=scaling_modifier,
#         nVoxel_x=int(nVoxel[0]),
#         nVoxel_y=int(nVoxel[1]),
#         nVoxel_z=int(nVoxel[2]),
#         sVoxel_x=float(sVoxel[0]),
#         sVoxel_y=float(sVoxel[1]),
#         sVoxel_z=float(sVoxel[2]),
#         center_x=float(center[0]),
#         center_y=float(center[1]),
#         center_z=float(center[2]),
#         prefiltered=False,
#         debug=pipe.debug,
#     )
#     voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

#     means3D = pc.get_xyz
#     means3D1 = pc.get_xyz

#     if Poses != 'None':
#         r = Poses[...,:3]
#         t = Poses[...,3]
#         R_inv = r.transpose(1, 2)
#         r4 = rotation_matrices_to_quaternions(R_inv)
#         n = t.shape[0]
#         n1 = means3D.shape[0]
#         means3D = means3D.repeat(n, 1, 1)
        
#         t = t.unsqueeze(1).repeat(1, n1, 1)
#         R_inv = R_inv.unsqueeze(1).repeat(1, n1, 1,1)
        
#         means3D = means3D+80  - t
#         means3D = means3D.unsqueeze(3)
#         means3D = R_inv @ means3D
#         means3D = means3D.squeeze(3) -80
#     density = pc.get_density

#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation
#     if Poses != 'None':
#         rotations = rotations.repeat(n, 1, 1)
#         r4 = r4.unsqueeze(1).repeat(1, n1, 1)
#         rotations = quaternion_multiplys(r4,rotations)
#     vol_pred_list = []
#     if Poses != 'None':
#         vol_pred_list = []
#         for i in range(n):
#             vol_pred, radii = voxelizer(
#                 means3D=means3D[i],
#                 opacities=density,
#                 scales=scales, 
#                 rotations=rotations[i],
#                 cov3D_precomp=cov3D_precomp,)
#             vol_pred_list.append(vol_pred.unsqueeze(0))
#         vol_pred = torch.cat(vol_pred_list, dim=0)  
#         vol_pred = vol_pred.view(n, 160*160, 1)      
#     else:
#         vol_pred, radii = voxelizer(
#             means3D=means3D,
#             opacities=density,
#             scales=scales,
#             rotations=rotations,
#             cov3D_precomp=cov3D_precomp,
#         )

#     return {
#         "vol": vol_pred,
#         "viewspace_points": means3D1,
#         "visibility_filter": radii > 0,
#         "radii": radii,
#     }

def quaternion_multiplys(q, r):
    """
    Multiply two sets of quaternions q and r.
    q: (90, 1000, 4) quaternion tensor
    r: (90, 1000, 4) quaternion tensor
    Returns:
    Result: (90, 1000, 4) quaternion tensor representing the product of each (q, r) pair
    """
    # Extract w, x, y, z components for both quaternions
    w0, x0, y0, z0 = q.unbind(-1)
    w1, x1, y1, z1 = r.unbind(-1)

    # Compute quaternion multiplication result using optimized operations
    result = torch.empty_like(q)
    result[..., 0] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    result[..., 1] = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    result[..., 2] = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    result[..., 3] = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    return result

def rotation_matrices_to_quaternions(r_matrices):
    """
    Convert a batch of rotation matrices to quaternions.
    r_matrices: (N, 3, 3) rotation matrices tensor
    Returns:
    quaternions: (N, 4) quaternions tensor
    """
    N = r_matrices.shape[0]
    quaternions = torch.zeros((N, 4), device=r_matrices.device)

    trace = r_matrices[:, 0, 0] + r_matrices[:, 1, 1] + r_matrices[:, 2, 2]
    mask_trace_positive = trace > 0

    # Handle trace > 0 case
    s = torch.sqrt(trace[mask_trace_positive] + 1.0 + 1e-6) * 0.5
    quaternions[mask_trace_positive, 0] = s
    quaternions[mask_trace_positive, 1] = (r_matrices[mask_trace_positive, 2, 1] - r_matrices[mask_trace_positive, 1, 2]) / (4.0 * s)
    quaternions[mask_trace_positive, 2] = (r_matrices[mask_trace_positive, 0, 2] - r_matrices[mask_trace_positive, 2, 0]) / (4.0 * s)
    quaternions[mask_trace_positive, 3] = (r_matrices[mask_trace_positive, 1, 0] - r_matrices[mask_trace_positive, 0, 1]) / (4.0 * s)

    # Handle trace <= 0 cases
    mask_col0 = (r_matrices[:, 0, 0] > r_matrices[:, 1, 1]) & (r_matrices[:, 0, 0] > r_matrices[:, 2, 2])
    mask_col1 = (r_matrices[:, 1, 1] > r_matrices[:, 2, 2]) & ~mask_col0
    mask_col2 = ~(mask_col0 | mask_col1)

    # Handle r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2] case
    s_col0 = torch.sqrt(1.0 + r_matrices[mask_col0, 0, 0] - r_matrices[mask_col0, 1, 1] - r_matrices[mask_col0, 2, 2]) * 2.0
    quaternions[mask_col0, 0] = (r_matrices[mask_col0, 2, 1] - r_matrices[mask_col0, 1, 2]) / s_col0
    quaternions[mask_col0, 1] = 0.25 * s_col0
    quaternions[mask_col0, 2] = (r_matrices[mask_col0, 0, 1] + r_matrices[mask_col0, 1, 0]) / s_col0
    quaternions[mask_col0, 3] = (r_matrices[mask_col0, 0, 2] + r_matrices[mask_col0, 2, 0]) / s_col0

    # Handle r[1, 1] > r[2, 2] case
    s_col1 = torch.sqrt(1.0 + r_matrices[mask_col1, 1, 1] - r_matrices[mask_col1, 0, 0] - r_matrices[mask_col1, 2, 2]) * 2.0
    quaternions[mask_col1, 0] = (r_matrices[mask_col1, 0, 2] - r_matrices[mask_col1, 2, 0]) / s_col1
    quaternions[mask_col1, 1] = (r_matrices[mask_col1, 0, 1] + r_matrices[mask_col1, 1, 0]) / s_col1
    quaternions[mask_col1, 2] = 0.25 * s_col1
    quaternions[mask_col1, 3] = (r_matrices[mask_col1, 1, 2] + r_matrices[mask_col1, 2, 1]) / s_col1

    # Handle remaining case (r[2, 2] is the largest)
    s_col2 = torch.sqrt(1.0 + r_matrices[mask_col2, 2, 2] - r_matrices[mask_col2, 0, 0] - r_matrices[mask_col2, 1, 1]) * 2.0
    quaternions[mask_col2, 0] = (r_matrices[mask_col2, 1, 0] - r_matrices[mask_col2, 0, 1]) / s_col2
    quaternions[mask_col2, 1] = (r_matrices[mask_col2, 0, 2] + r_matrices[mask_col2, 2, 0]) / s_col2
    quaternions[mask_col2, 2] = (r_matrices[mask_col2, 1, 2] + r_matrices[mask_col2, 2, 1]) / s_col2
    quaternions[mask_col2, 3] = 0.25 * s_col2

    # Normalize quaternions
    norms = quaternions.norm(dim=1, keepdim=True) + 1e-6
    quaternions = quaternions / norms
    
    return quaternions

# def quaternions_to_rotation_matrices(quaternions):
#     """
#     Convert a batch of quaternions to rotation matrices.
#     quaternions: (N, 4) quaternions tensor
#     Returns:
#     r_matrices: (N, 3, 3) rotation matrices tensor
#     """
#     N = quaternions.shape[0]
#     r_matrices = torch.zeros((N, 3, 3), device=quaternions.device)

#     w, x, y, z = quaternions.unbind(-1)

#     r_matrices[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
#     r_matrices[:, 0, 1] = 2 * (x * y - z * w)
#     r_matrices[:, 0, 2] = 2 * (x * z + y * w)

#     r_matrices[:, 1, 0] = 2 * (x * y + z * w)
#     r_matrices[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
#     r_matrices[:, 1, 2] = 2 * (y * z - x * w)

#     r_matrices[:, 2, 0] = 2 * (x * z - y * w)
#     r_matrices[:, 2, 1] = 2 * (y * z + x * w)
#     r_matrices[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

#     return r_matrices

def quaternions_to_rotation_matrices(quaternions):
    """
    Convert a batch of quaternions to rotation matrices.
    quaternions: (N, 4) quaternions tensor
    Returns:
    r_matrices: (N, 3, 3) rotation matrices tensor
    """
    # Normalize the quaternions to prevent numerical instability
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    w, x, y, z = quaternions.unbind(-1)

    # Use broadcasting for efficient calculation of rotation matrices
    r_matrices = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),       1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(-1, 3, 3)

    return r_matrices

def query(pc: GaussianModel, center, nVoxel, sVoxel, quaternion, trans,pipe, scaling_modifier=1.0):
    # Precompute GaussianVoxelizationSettings
    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    # Fetch necessary data once to reduce repeated function calls
    means3D = pc.get_xyz
    #means3D1 = pc.get_xyz
    density = pc.get_density

    # Initialize transformations if Poses are provided
    if quaternion != 'None':
        # r = Poses[..., :3]
        # t = Poses[..., 3]
        # R_inv = r.transpose(1, 2)
        #print(quaternion.shape)
        r4 = quaternions_to_rotation_matrices(quaternion)

        #r4 = rotation_matrices_to_quaternions(R_inv)
        n = trans.shape[0]
        n1 = means3D.shape[0]

        # Efficiently compute means3D in batch
        means3D = means3D.repeat(n, 1, 1)
        t = trans.unsqueeze(1).repeat(1, n1, 1)
        R_inv = r4.unsqueeze(1).repeat(1, n1, 1, 1)

        # Avoid inplace operations to prevent CUDA issues
        means3D = (means3D + 80 - t).unsqueeze(3)
        means3D = torch.matmul(R_inv, means3D).squeeze(3) - 80

    # Precompute covariance or scaling/rotation based on the pipeline configuration
    scales, rotations, cov3D_precomp = None, None, None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Update rotations if Poses are provided
    if quaternion != 'None':
        rotations = rotations.repeat(n, 1, 1)
        quaternion = quaternion.unsqueeze(1).repeat(1, n1, 1)
        rotations = quaternion_multiplys(quaternion, rotations)

    # Use batch processing to reduce loop overhead and improve GPU utilization
    if quaternion != 'None':
        vol_pred_list = []
        for i in range(n):
            vol_pred, radii = voxelizer(
                means3D=means3D[i],
                opacities=density,
                scales=scales,
                rotations=rotations[i],
                cov3D_precomp=cov3D_precomp,
            )
            vol_pred_list.append(vol_pred.unsqueeze(0))
        vol_pred = torch.cat(vol_pred_list, dim=0)
        vol_pred = vol_pred.view(n, 160 * 160, 1)
    else:
        vol_pred, radii = voxelizer(
            means3D=means3D,
            opacities=density,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    return {
        "vol": vol_pred,
        "viewspace_points": pc.get_xyz,  # Directly use original points
        "visibility_filter": radii > 0,
        "radii": radii,
    }