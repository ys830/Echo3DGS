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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="us_gaussian_voxelization",
    packages=["us_gaussian_voxelization"],
    ext_modules=[
        CUDAExtension(
            name="us_gaussian_voxelization._C",
            sources=[
                "cuda_voxelizer/voxelizer_impl.cu",
                "cuda_voxelizer/forward.cu",
                "cuda_voxelizer/backward.cu",
                "voxelize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"
                    )
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
