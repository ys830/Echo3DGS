import torch
import numpy as np
from utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
)
from torch import nn
import os
#from utils.system_utils import mkdir_p
from utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils import strip_symmetric, build_scaling_rotation
from utils import BasicPointCloud
#from apex.optimizers import FusedAdam
# class QuaternionModel(nn.Module):
#     def __init__(self, initial_quaternions):
#         super(QuaternionModel, self).__init__()
#         # Convert initial_quaternions to a learnable parameter
#         self.RR = nn.Parameter(torch.tensor(initial_quaternions, dtype=torch.float32),requires_grad=False)
#         #self.RR = nn.Parameter(torch.tensor(initial_quaternions, dtype=torch.float32))

#     def forward(self):
#         # Ensure that quaternions are normalized to stay valid
#         norm = torch.norm(self.RR, dim=1, keepdim=True)
#         normalized_quaternions = self.RR / norm
#         return normalized_quaternions

# class TranslationModel(nn.Module):
#     def __init__(self, initial_translation):
#         super(TranslationModel, self).__init__()
#         # Convert initial_translation to a learnable parameter
#         self.TT = nn.Parameter(torch.tensor(initial_translation, dtype=torch.float32),requires_grad=False)
#         #self.TT = nn.Parameter(torch.tensor(initial_translation, dtype=torch.float32))

#     def forward(self):
#         # Simply return the learnable translation parameter
#         return self.TT

class QuaternionModel(nn.Module):
    def __init__(self, initial_quaternions):
        super(QuaternionModel, self).__init__()
        # Convert initial_quaternions to a fixed tensor
        #self.RR_initial = torch.tensor(initial_quaternions, dtype=torch.float32)
        self.register_buffer('RR_initial', torch.tensor(initial_quaternions, dtype=torch.float32))
        # Learnable increment to be added to the initial quaternions
        self.RR = nn.Parameter(torch.zeros_like(self.RR_initial))  # Initialize with zeros

    def forward(self):
        # Add the learnable increment to the initial quaternions
        quaternions = self.RR_initial + self.RR
        # Ensure that quaternions are normalized to stay valid
        norm = torch.norm(quaternions, dim=1, keepdim=True)
        normalized_quaternions = quaternions / norm
        return normalized_quaternions
    
class TranslationModel(nn.Module):
    def __init__(self, initial_translation):
        super(TranslationModel, self).__init__()
        # Convert initial_translation to a fixed tensor
        #self.TT_initial = torch.tensor(initial_translation, dtype=torch.float32)
        self.register_buffer('TT_initial', torch.tensor(initial_translation, dtype=torch.float32))
        # Learnable increment to be added to the initial translation
        self.TT = nn.Parameter(torch.zeros_like(self.TT_initial))  # Initialize with zeros

    def forward(self):
        # Add the learnable increment to the initial translation
        translation = self.TT_initial + self.TT
        return translation

class GaussianModel:
    def setup_functions(self, scale_bound):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        scale_min_bound, scale_max_bound = scale_bound
        self.scaling_activation = (
            lambda x: torch.sigmoid(x) * scale_max_bound + scale_min_bound
        )
        self.scaling_inverse_activation = lambda x: inverse_sigmoid(
            torch.relu((x - scale_min_bound) / scale_max_bound) + 1e-8
        )

        self.covariance_activation = build_covariance_from_scaling_rotation

        # self.density_activation = torch.sigmoid  # Origin is sigmoid
        # self.density_inverse_activation = inverse_sigmoid
        self.density_activation = torch.nn.Softplus()
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    #def __init__(self, scale_bound,quaternion_model,translation_model):
    def __init__(self, scale_bound):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        # self.translation_model = translation_model  # Store TranslationModel instance
        # self.quaternion_model = quaternion_model    # Store QuaternionModel instance

        self.setup_functions(scale_bound)

    def capture(self):
        return (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # self.translation_model,
            # self.quaternion_model 
        )

    def restore(self, model_args, training_args):
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            # self.translation_model,
            # self.quaternion_model,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, pcd, spatial_lr_scale: float, init_from="pcd"):
        self.spatial_lr_scale = spatial_lr_scale
        #print(init_from)
        if init_from in ["pcd", "random", "unifrm"]:
            if init_from == "pcd":
                #fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
                fused_point_cloud = torch.tensor(np.asarray(pcd[:,:3])).float().cuda()
                print(
                    "Initialize gaussians from {} estimated points".format(
                        fused_point_cloud.shape[0]
                    )
                )
                # fused_density = (
                #     self.density_inverse_activation(
                #         torch.tensor(np.asarray(pcd.colors[:, :1]))
                #     )
                #     .float()
                #     .cuda()
                # )
                fused_density = torch.tensor(np.asarray(pcd[:,10:11])).float().cuda()
                rots = torch.tensor(np.asarray(pcd[:,3:7])).float().cuda()
                scales = torch.tensor(np.asarray(pcd[:,7:10])).float().cuda()

            elif init_from == "random":
                fused_point_cloud = pcd[:, :-1].float().cuda()
                fused_density = self.density_inverse_activation(
                    pcd[:, -1:].float().cuda()
                )
                print(
                    "Initialize gaussians from {} random points".format(
                        fused_point_cloud.shape[0]
                    )
                )
                dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd[:, 0:3])).float().cuda()), 0.0000001)
                scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3)
                rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots[:, 0] = 1
                
            elif init_from == "unifrm":
                #print(11111)
                fused_point_cloud = pcd[:, :-1].float().cuda()
                fused_density = self.density_inverse_activation(
                    pcd[:, -1:].float().cuda()
                )
                print(
                    "Initialize gaussians from {} random points".format(
                        fused_point_cloud.shape[0]
                    )
                )

            # dist2 = torch.clamp_min(
            #     distCUDA2(fused_point_cloud),
            #     self.scale_bound[0] ** 2,
            # )
            # scales = self.scaling_inverse_activation(torch.sqrt(dist2))[
            #     ..., None
            # ].repeat(1, 3)
            # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            # rots[:, 0] = 1
                
      

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        elif init_from == "one_point":
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        elif init_from == "pickle":
            fused_point_cloud = torch.tensor(pcd["position"]).float().cuda()
            scales = self.scaling_inverse_activation(
                torch.tensor(pcd["scale"]).float().cuda()
            )
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1
            fused_density = self.density_inverse_activation(
                torch.tensor(pcd["density"]).float().cuda()
            )

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        else:
            raise ValueError("Unsupported initialization mode!")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
            # {
            #     "params": [self.quaternion_model.RR],  # QuaternionModel params
            #     "lr": training_args.dR_lr_init * self.spatial_lr_scale,
            #     "name": "quaternion",
            # },
            # {
            #     "params": [self.translation_model.TT],  # TranslationModel params
            #     "lr": training_args.dT_lr_init * self.spatial_lr_scale,
            #     "name": "translation",
            # },
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )
        # self.quaternion_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.dR_lr_init * self.spatial_lr_scale,
        #     lr_final=training_args.dR_lr_final * self.spatial_lr_scale,
        #     max_steps=training_args.dR_lr_max_steps,
        # )
        # self.translation_scheduler_args = get_expon_lr_func(
        #     lr_init=training_args.dT_lr_init * self.spatial_lr_scale,
        #     lr_final=training_args.dT_lr_final * self.spatial_lr_scale,
        #     max_steps=training_args.dT_lr_max_steps,
        # )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr
            # if param_group["name"] == "quaternion":
            #     lr = self.quaternion_scheduler_args(iteration)
            #     param_group["lr"] = lr
            # if param_group["name"] == "translation":
            #     lr = self.translation_scheduler_args(iteration)
            #     param_group["lr"] = lr
            

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        densities = self._density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, densities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)
    
    def save_reptail_npy(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        densities = self._density.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, densities, scale, rotation), axis=1)
        np.save(path,attributes)
    

    def reset_density(self, reset_density=0.1):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        densities = np.asarray(plydata.elements[0]["density"])[..., np.newaxis]

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._density = nn.Parameter(
            torch.tensor(densities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ["quaternion", "translation"]:
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # if group["name"] not in tensors_dict:
            #     print(f"Warning: {group['name']} not found in tensors_dict")
            #     continue
            if group["name"] in ["quaternion", "translation"]:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            # "quaternion": self.quaternion_model.RR,
            # "translation": self.translation_model.TT
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self.quaternion_model.RR = optimizable_tensors["quaternion"]
        # self.translation_model.TT = optimizable_tensors["translation"]
    

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

    def densify_and_prune(
        self,
        grads,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        # grads = self.xyz_gradient_accum / self.denom
        # grads[grads.isnan()] = 0.0

        if max_num_gaussians is not None:
            if grads.shape[0] < max_num_gaussians and densify_scale_threshold:
                self.densify_and_clone(grads, max_grad, densify_scale_threshold)
                self.densify_and_split(grads, max_grad, densify_scale_threshold)

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        #print(viewspace_point_tensor.grad)
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :3], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
