## Requirement:
einops==0.4.1
kornia==0.6.5
pytorch-lightning==1.7.7
matplotlib==3.5.2
opencv-python==4.6.0.66
imageio
scipy

pip install -e submodules/us_gaussian_voxelization
pip install -e submodules/simple-knn


## Training

CUDA_VISIBLE_DEVICES=0 python train_heart_gs_torch.py --root_dir './3d_slice/012' --exp_name './GS/3d012/'
