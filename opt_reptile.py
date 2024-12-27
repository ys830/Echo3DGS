import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--root_dir', type=str, default='./3d_slice/012',
                        help='root directory of dataset')
    parser.add_argument('--root_vol', type=str, default='./ini_meta/3d_slice/005_mlp5.npy',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='heart',
                        choices=['heart', 'heart_dy','heart_dy_3d'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    # training options
    parser.add_argument('--batch_size', type=int, default=8100, ##8192
                        help='number of rays in a batch')
    parser.add_argument('--num_epochs', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--T', type=float, default=4,
                        help='Length of cardiac cycle')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2, #1e-2 dy:1e-4
                        help='learning rate')

    # validation options
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='./GS/3d012_2/',
                        help='experiment name')
    parser.add_argument('--path_ply', type=str, default='./ini_pcd/fused_point_cloud_array.npy',
                        help='ini_ply_path')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    parser.add_argument('--scale_min_bound', type=float, default=0.1,
                        help='min_bound')
    parser.add_argument('--scale_max_bound', type=int, default=80,
                        help='max_bound')
    parser.add_argument('--sVoxel', type=int, default=160,
                        help='size of voxel') 
    parser.add_argument('--nVoxel', type=int, default=160,
                        help='number of voxel')
    parser.add_argument('--model_path', type=str, default='',
                        help='model_path')
    parser.add_argument('--init_from', type=str, default='random_2000', ##random_2000
                        help='model_path')
    parser.add_argument('--densify_scale_threshold', type=float, default=0.05,
                        help='densify_scale_threshold')
    parser.add_argument('--densify_until_iter', type=int, default=15000,
                        help='densify_until_iter')
    parser.add_argument('--densify_from_iter', type=int, default=10,
                        help='densify_from_iter')
    parser.add_argument('--densify_grad_threshold', type=float, default=5.0e-5,##5.0e-5
                        help='densify_grad_threshold')
    parser.add_argument('--density_min_threshold', type=float, default=0.00001,##0.00001
                        help='density_min_threshold')
    parser.add_argument('--max_num_gaussians', type=int, default=500000,
                        help='max_num_gaussians')
    parser.add_argument('--opacity_reset_interval', type=int, default=50,
                        help='opacity_reset_interval')
    parser.add_argument('--max_screen_size', type=str, required=None,
                        help='max_screen_size')
    parser.add_argument('--max_scale', type=str, required=None,
                        help='max_scale')
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--bbox", type=eval, default=[[-80,-80,-80],[80,80,80]], help="A list of lists of coordinates")
    parser.add_argument('--tv_vol_size', type=int, default=160,
                        help='tv_vol_size')
    parser.add_argument('--compute_cov3D_python', action='store_true', default=False,
                        help='compute_cov3D_python')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug')
    
    parser.add_argument('--position_lr_init', type=float, default=0.1,  ###0.0002
                        help='position_lr_init')
    parser.add_argument('--position_lr_final', type=float, default=0.01,###0.00002
                        help='position_lr_final')
    parser.add_argument('--position_lr_max_steps', type=int, default=30000,
                        help='position_lr_max_steps')
    
    parser.add_argument('--density_lr_init', type=float, default=0.1, ###0.01
                        help='position_lr_init')
    parser.add_argument('--density_lr_final', type=float, default=0.01, ###0.001
                        help='position_lr_final')
    parser.add_argument('--density_lr_max_steps', type=int, default=30000,
                        help='position_lr_max_steps')
    
    parser.add_argument('--scaling_lr_init', type=float, default=0.1,###0.005
                        help='position_lr_init')
    parser.add_argument('--scaling_lr_final', type=float, default=0.01,###0.0005
                        help='position_lr_final')
    parser.add_argument('--scaling_lr_max_steps', type=int, default=30000,
                        help='position_lr_max_steps')
    
    parser.add_argument('--rotation_lr_init', type=float, default=0.1,###0.001
                        help='position_lr_init')
    parser.add_argument('--rotation_lr_final', type=float, default=0.01,###0.0001
                        help='position_lr_final')
    parser.add_argument('--rotation_lr_max_steps', type=int, default=30000,
                        help='position_lr_max_steps')
    
    parser.add_argument('--dR_lr_init', type=float, default=0.001,
                        help='dR_lr_init')
    parser.add_argument('--dR_lr_final', type=float, default=0.0001,
                        help='dR_lr_final')
    parser.add_argument('--dR_lr_max_steps', type=int, default=30000,
                        help='dR_lr_max_steps')
    
    parser.add_argument('--dT_lr_init', type=float, default=0.001,
                        help='dT_lr_init')
    parser.add_argument('--dT_lr_final', type=float, default=0.0001,
                        help='dT_lr_final')
    parser.add_argument('--dT_lr_max_steps', type=int, default=30000,
                        help='dT_lr_max_steps')
    
    parser.add_argument('--densification_interval', type=int, default=50,
                        help='densification_interval')
    
    # parser.add_argument('--N', type=int, default=120,
    #                     help='N')


    return parser.parse_args()
