import torch

# 定义四元数到旋转矩阵的转换函数
def quaternions_to_rotation_matrices(quaternions):
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)
    w, x, y, z = quaternions.unbind(-1)
    
    r_matrices = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),       1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(-1, 3, 3)
    
    return r_matrices

# 定义旋转矩阵到四元数的转换函数
def rotation_matrices_to_quaternions(r_matrices):
    N = r_matrices.shape[0]
    quaternions = torch.zeros((N, 4), device=r_matrices.device)
    trace = r_matrices[:, 0, 0] + r_matrices[:, 1, 1] + r_matrices[:, 2, 2]
    mask_trace_positive = trace > 0
    
    s = torch.sqrt(trace[mask_trace_positive] + 1.0 + 1e-6) * 0.5
    quaternions[mask_trace_positive, 0] = s
    quaternions[mask_trace_positive, 1] = (r_matrices[mask_trace_positive, 2, 1] - r_matrices[mask_trace_positive, 1, 2]) / (4.0 * s)
    quaternions[mask_trace_positive, 2] = (r_matrices[mask_trace_positive, 0, 2] - r_matrices[mask_trace_positive, 2, 0]) / (4.0 * s)
    quaternions[mask_trace_positive, 3] = (r_matrices[mask_trace_positive, 1, 0] - r_matrices[mask_trace_positive, 0, 1]) / (4.0 * s)

    mask_col0 = (r_matrices[:, 0, 0] > r_matrices[:, 1, 1]) & (r_matrices[:, 0, 0] > r_matrices[:, 2, 2])
    mask_col1 = (r_matrices[:, 1, 1] > r_matrices[:, 2, 2]) & ~mask_col0
    mask_col2 = ~(mask_col0 | mask_col1)
    
    s_col0 = torch.sqrt(1.0 + r_matrices[mask_col0, 0, 0] - r_matrices[mask_col0, 1, 1] - r_matrices[mask_col0, 2, 2]) * 2.0
    quaternions[mask_col0, 0] = (r_matrices[mask_col0, 2, 1] - r_matrices[mask_col0, 1, 2]) / s_col0
    quaternions[mask_col0, 1] = 0.25 * s_col0
    quaternions[mask_col0, 2] = (r_matrices[mask_col0, 0, 1] + r_matrices[mask_col0, 1, 0]) / s_col0
    quaternions[mask_col0, 3] = (r_matrices[mask_col0, 0, 2] + r_matrices[mask_col0, 2, 0]) / s_col0

    s_col1 = torch.sqrt(1.0 + r_matrices[mask_col1, 1, 1] - r_matrices[mask_col1, 0, 0] - r_matrices[mask_col1, 2, 2]) * 2.0
    quaternions[mask_col1, 0] = (r_matrices[mask_col1, 0, 2] - r_matrices[mask_col1, 2, 0]) / s_col1
    quaternions[mask_col1, 1] = (r_matrices[mask_col1, 0, 1] + r_matrices[mask_col1, 1, 0]) / s_col1
    quaternions[mask_col1, 2] = 0.25 * s_col1
    quaternions[mask_col1, 3] = (r_matrices[mask_col1, 1, 2] + r_matrices[mask_col1, 2, 1]) / s_col1

    s_col2 = torch.sqrt(1.0 + r_matrices[mask_col2, 2, 2] - r_matrices[mask_col2, 0, 0] - r_matrices[mask_col2, 1, 1]) * 2.0
    quaternions[mask_col2, 0] = (r_matrices[mask_col2, 1, 0] - r_matrices[mask_col2, 0, 1]) / s_col2
    quaternions[mask_col2, 1] = (r_matrices[mask_col2, 0, 2] + r_matrices[mask_col2, 2, 0]) / s_col2
    quaternions[mask_col2, 2] = (r_matrices[mask_col2, 1, 2] + r_matrices[mask_col2, 2, 1]) / s_col2
    quaternions[mask_col2, 3] = 0.25 * s_col2

    norms = quaternions.norm(dim=1, keepdim=True) + 1e-6
    quaternions = quaternions / norms    
    return quaternions

# 生成随机的四元数
quaternions = torch.randn(10, 4)
quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

# 将四元数转换为旋转矩阵，再转换回四元数
r_matrices = quaternions_to_rotation_matrices(quaternions)
reconstructed_quaternions = rotation_matrices_to_quaternions(r_matrices)

# 考虑正负号的差异，计算误差
error = torch.min(
    torch.norm(quaternions - reconstructed_quaternions, dim=-1),
    torch.norm(quaternions + reconstructed_quaternions, dim=-1)
)

print(error)