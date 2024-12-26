#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
import os.path as osp
import torch
import pickle
from utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
#from arguments import ModelParams
#from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from scene.dataset_readers import fetchPly


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args,
        gaussians: GaussianModel,
        load_iteration=None,
        init_from="pcd",
        ply_path="pcd",
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if self.loaded_iter:
            # ! Need to change
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            if init_from[:6] == "random":
                n_point = int(init_from[7:])
                assert n_point > 0, "Specify valid number of random points"
                bbox = torch.tensor(args.bbox)
                pcd = bbox[0] + (bbox[1] - bbox[0]) * torch.rand([n_point, 3])
                density = torch.rand([n_point, 1]) * 0.1
                point_cloud = torch.concat([pcd, density], dim=-1)
                init_from = "random"
            elif init_from[:6] == "unifrm":
                n_point = int(init_from[7:])
                assert n_point > 0, "Specify valid number of random points"
                bbox = torch.tensor(args.bbox)
                n_points_per_dim = int(np.cbrt(n_point))
                assert n_points_per_dim**3 == n_point, "n_point must be a perfect cube"
                x = torch.linspace(bbox[0, 0], bbox[1, 0], n_points_per_dim)
                y = torch.linspace(bbox[0, 1], bbox[1, 1], n_points_per_dim)
                z = torch.linspace(bbox[0, 2], bbox[1, 2], n_points_per_dim)

                # 使用网格生成均匀的点云
                xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
                pcd = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

                # 生成随机密度
                density = torch.rand([n_point, 1]) * 0.1
               
                # 拼接点云和密度
                point_cloud = torch.cat([pcd, density], dim=-1)
                init_from = "unifrm"
            elif init_from == "pcd":
                #point_cloud = fetchPly(ply_path)
                point_cloud = np.load(ply_path)
                print(f"Initialize gaussians with pcd {ply_path}")
            elif init_from == "pickle":
                with open(ply_path[:-3] + "pickle", "rb") as handle:
                    point_cloud = pickle.load(handle)
            else:
                point_cloud = None

            self.gaussians.create_from_pcd(point_cloud, 1.0, init_from)

    def save(self, path,iteration, queryfunc, tv_vol_center,tv_vol_nVoxel,tv_vol_sVoxel,pipe):
        # point_cloud_path = osp.join(
        #     self.model_path, "point_cloud/iteration_{}".format(iteration)
        # )
        point_cloud_path = osp.join(
            path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(osp.join(point_cloud_path, "point_cloud.ply"))

        # Save volume
        #scanner_cfg = self.meta_data["scanner"]
        # query_pkg = queryfunc(
        #     self.gaussians,
        #     scanner_cfg["offOrigin"],
        #     scanner_cfg["nVoxel"],
        #     scanner_cfg["sVoxel"],
        #     pipe,
        # )
        query_pkg = queryfunc(
            self.gaussians,
            tv_vol_center,
            tv_vol_nVoxel,
            tv_vol_sVoxel,
            'None',
            'None',
            pipe,
        )
        vol_pred = query_pkg["vol"].clip(0.0, 1.0)
        #vol_gt = self.vol_gt.clip(0.0, 1.0)

        #np.save(osp.join(point_cloud_path, "vol_gt.npy"), vol_gt.detach().cpu().numpy())
        np.save(
            osp.join(point_cloud_path, "vol_pred.npy"), vol_pred.detach().cpu().numpy()
        )

    # def getTrainCameras(self):
    #     return self.train_cameras

    # def getTestCameras(self):
    #     return self.test_cameras
