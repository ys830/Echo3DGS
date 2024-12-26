import torch
import numpy as np
from kornia import create_meshgrid
from kornia.utils import create_meshgrid3d
from einops import rearrange


#@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions_3d(D,H, W, device='cpu', flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid3d(D, H, W, False, device=device)[0] # (H, W, 2)
    grid = grid/160
    i, j, k = grid.unbind(-1)

    directions = torch.stack([i,k,j], -1)

    if flatten:
        directions = directions.reshape(-1, 3)
        directions = directions.requires_grad_(True)
    return directions

#@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, device='cpu', flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    #print(grid)
    i, j = grid.unbind(-1)

    directions = \
            torch.stack([i,j,80*torch.ones_like(i)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
    return directions

#@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (N, 3) ray directions in camera coordinate
        c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinate
        rays_d: (N, 3), the direction of the rays in world coordinate
    """
    rays_r = rearrange(directions, 'n c -> n 1 c') @ \
                 rearrange(c2w[..., :3], 'n a b -> n b a')
    rays_r = rearrange(rays_r, 'n 1 c -> n c')
    # print(c2w.shape)
    # print('directions',directions.shape)
    #print('rays_r',rays_r.shape)
    rays_t = c2w[..., 3].expand_as(rays_r)
    #print('rays_t',rays_t.shape)
    rays_d = rays_r + rays_t
    #rays_d = rays_d/160-0.5 #ori
    rays_d = rays_d/80-1 #gs
    #rays_d = rays_d/160
    return rays_d


#@torch.cuda.amp.autocast(dtype=torch.float32)
def axisangle_to_R(v):
    """
    Convert an axis-angle vector to rotation matrix
    from https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py#L47

    Inputs:
        v: (3) or (B, 3)
    
    Outputs:
        R: (3, 3) or (B, 3, 3)
    """
    v_ndim = v.ndim
    if v_ndim==1:
        v = rearrange(v, 'c -> 1 c')
    zero = torch.zeros_like(v[:, :1]) # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1) # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1) # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1)+1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v)/norm_v)*skew_v + \
        ((1-torch.cos(norm_v))/norm_v**2)*(skew_v@skew_v)
    
    if torch.isnan(R).any() or torch.isinf(R).any():
        print("NaN or Inf found in R matrix")
        raise ValueError("NaN or Inf detected in rotation matrix")
    if v_ndim==1:
        R = rearrange(R, '1 c d -> c d')
    return R

#@torch.cuda.amp.autocast(dtype=torch.float32)
def R2axangle(R):

    theta = torch.acos((R.trace()-1)/2)
    if theta == 0:
        w = torch.tensor([0,0,0])
    else:
        aix1 = R[2,1]-R[1,2]
        aix2 = R[0,2]-R[2,0]
        aix3 = R[1,0]-R[0,1]
        aix = torch.tensor([aix1,aix2,aix3])
        w = theta * (1 / (2 * torch.sin(theta))) * aix

    return w

#@torch.cuda.amp.autocast(dtype=torch.float32)
def q_to_R(q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        #print(q.shape)
        qa,qb,qc,qd = q.unbind(dim=-1)
        #print(qa.shape)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        #print(R.shape)
        return R

#@torch.cuda.amp.autocast(dtype=torch.float32)
def R_to_q(R,eps=1e-8): # [B,3,3]
    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # FIXME: this function seems a bit problematic, need to double-check
    row0,row1,row2 = R.unbind(dim=-2)
    R00,R01,R02 = row0.unbind(dim=-1)
    R10,R11,R12 = row1.unbind(dim=-1)
    R20,R21,R22 = row2.unbind(dim=-1)
    t = R[...,0,0]+R[...,1,1]+R[...,2,2]
    r = (1+t+eps).sqrt()
    qa = 0.5*r
    qb = (R21-R12).sign()*0.5*(1+R00-R11-R22+eps).sqrt()
    qc = (R02-R20).sign()*0.5*(1-R00+R11-R22+eps).sqrt()
    qd = (R10-R01).sign()*0.5*(1-R00-R11+R22+eps).sqrt()
    q = torch.stack([qa,qb,qc,qd],dim=-1)
    for i,qi in enumerate(q):
        if torch.isnan(qi).any():
            K = torch.stack([torch.stack([R00-R11-R22,R10+R01,R20+R02,R12-R21],dim=-1),
                                torch.stack([R10+R01,R11-R00-R22,R21+R12,R20-R02],dim=-1),
                                torch.stack([R20+R02,R21+R12,R22-R00-R11,R01-R10],dim=-1),
                                torch.stack([R12-R21,R20-R02,R01-R10,R00+R11+R22],dim=-1)],dim=-2)/3.0
            K = K[i]
            eigval,eigvec = torch.linalg.eigh(K)
            V = eigvec[:,eigval.argmax()]
            q[i] = torch.stack([V[3],V[0],V[1],V[2]])
    return q
def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses, pts3d=None):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered

    return poses_centered

def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,2*mean_h],
            [0,0,1,-t]
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th)],
            [0,1,0],
            [np.sin(th),0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius)]
    return np.stack(spheric_poses, 0)

def slice_operator(points, vol):
   
    sample_pos = points.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    vol = vol.unsqueeze(0).unsqueeze(0)
    #print('sample_pos.shape',sample_pos.shape)
    slice_pred = torch.nn.functional.grid_sample(vol, sample_pos, mode='bilinear',padding_mode='zeros', align_corners=False)
    slice_pred = slice_pred.squeeze(0).squeeze(0).squeeze(0).squeeze(0).unsqueeze(1)
    return slice_pred