import torch
from torch import nn
from opt import get_opts
import os
import imageio
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays,slice_operator
from ssimloss import S3IM
from losses import NeRFLoss
from losses_tv import tv_3d_loss
from torchmetrics import PeakSignalNoiseRatio
from gaussian_renderer import query,rotation_matrices_to_quaternions
from scene import GaussianModel,Scene
from scene.gaussian_model import QuaternionModel,TranslationModel
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

import warnings; warnings.filterwarnings("ignore")

class GSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        #self.loss = S3IM(kernel_size=4, stride=4, repeat_time=10, patch_height=160, patch_width=160)
        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.bbox = torch.tensor(hparams.bbox)
        sVoxel = torch.tensor(hparams.sVoxel)
        nVoxel = torch.tensor(hparams.nVoxel)
        dVoxel = (sVoxel / nVoxel)
        self.scale_min_bound = hparams.scale_min_bound * float(dVoxel.min())
        self.scale_max_bound = hparams.scale_max_bound * float(dVoxel.min())
    
        self.densify_scale_threshold = (
            hparams.densify_scale_threshold * float(sVoxel.min())
            if hparams.densify_scale_threshold
            else None
        )
        tv_vol_size = hparams.tv_vol_size
        self.tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        self.tv_vol_nVoxel1 = torch.tensor([1,tv_vol_size,tv_vol_size])
        
        self.tv_vol_sVoxel = (sVoxel / nVoxel * self.tv_vol_nVoxel)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        fixed_poses = self.train_dataset.poses
        self.test_dataset = dataset(split='test', **kwargs)
        self.fixed_images = self.train_dataset.rays
        self.initial_translation = fixed_poses[...,3].cuda()
        #print('aaaaaaaa',self.device)
        initial_quaternion = fixed_poses[...,:3]
        initial_quaternion = initial_quaternion.transpose(1, 2)
        self.initial_quaternions = rotation_matrices_to_quaternions(initial_quaternion).cuda()
        #print(self.initial_quaternions.shape)
        # self.translation_model = TranslationModel(self.initial_translation)
        # self.quaternion_model = QuaternionModel(self.initial_quaternions)
        # self.translation_model=translation_model()
        # self.quaternion_model =quaternion_model()
        # print('AAA',self.translation_model)
        self.gaussians = GaussianModel([self.scale_min_bound, self.scale_max_bound])
        self.scene = Scene(
            hparams,
            self.gaussians,
            load_iteration=None,
            init_from=hparams.init_from,
            ply_path=hparams.path_ply,
        )
        self.gaussians.training_setup(hparams)

        if hparams.start_checkpoint:
            model_params, self.first_iter = torch.load(hparams.start_checkpoint)
            self.gaussians.restore(model_params, hparams)       


    def forward(self,batch,split):

        # quaternion = self.gaussians.quaternion_model()
        # trans = self.gaussians.translation_model()

        quaternion = self.initial_quaternions
        trans = self.initial_translation
        #print(trans.device)
        self.tv_vol_center = (self.bbox[0] + self.tv_vol_sVoxel / 2) + (
                self.bbox[1] - self.tv_vol_sVoxel - self.bbox[0]
            ) * torch.rand(3)

        vol_pkg = query(
                self.scene.gaussians,
                self.tv_vol_center,
                self.tv_vol_nVoxel1,
                self.tv_vol_sVoxel,
                quaternion,
                trans,
                self.hparams,
            )
        prediction, viewspace_point_tensor, visibility_filter, radii = (
            vol_pkg["vol"],
            vol_pkg["viewspace_points"],
            vol_pkg["visibility_filter"],
            vol_pkg["radii"],
        )
        vol_pred = query(
                self.scene.gaussians,
                self.tv_vol_center,
                self.tv_vol_nVoxel,
                self.tv_vol_sVoxel,
                'None',
                'None',
                self.hparams,
            )["vol"]

        return prediction,vol_pred,viewspace_point_tensor,visibility_filter,radii

    
    def configure_optimizers(self):
        # self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        # self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        opts = []
        self.net_opt = self.gaussians.optimizer

        opts += [self.net_opt]
        return opts

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=4,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def training_step(self,batch, batch_nb, *args):
        #torch.autograd.set_detect_anomaly(True)
        self.gaussians.update_learning_rate(self.global_step)
        results,vol_pred,self.viewspace_point_tensor,self.visibility_filter,self.radii = self(batch, split='train')
        loss_d = 0
        n = results.shape[0]
        for i in range(n):
            loss1 = self.loss(results[i], self.fixed_images[i].cuda())
            loss = sum(lo.mean() for lo in loss1.values())
            loss_d +=loss
        loss_d /= n
        total_loss = loss_d
        loss_tv = tv_3d_loss(vol_pred, reduction="mean")
        total_loss += 1 * loss_tv
        with torch.no_grad():
            psnr = 0
            for i in range(n):
                psnr += self.train_psnr(results[i], self.fixed_images[i].cuda())
            psnr /= n 
            # psnr += self.train_psnr(results[0], self.fixed_images[0].cuda())
        
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss_d)
        self.log('train/psnr',psnr, True)
        self.log('train/n_point',self.gaussians.get_xyz.shape[0], True)
        # save_dir = f'ckpts/{self.hparams.dataset_name}/{self.hparams.exp_name}/img/'
        # os.makedirs(save_dir, exist_ok=True)
        # result =results[0].detach().cpu().numpy()
        # img =(255*result.reshape(160, 160)).astype(np.uint8)
        # imageio.imwrite(f'{save_dir}/step_{self.global_step}.png', img)

        return total_loss
    
    def on_train_epoch_end(self):
        #if self.gaussians.get_xyz.shape[0] < 10000:
        with torch.no_grad():
            self.gaussians.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussians.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter]
            )
            self.gaussians.add_densification_stats(self.viewspace_point_tensor, self.visibility_filter)
            grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom
            grads[grads.isnan()] = 0.0

            if self.global_step < self.hparams.densify_until_iter:
                if (
                    self.global_step > self.hparams.densify_from_iter
                    and self.global_step % self.hparams.densification_interval == 0
                ):
                    self.gaussians.densify_and_prune(
                        grads,
                        self.hparams.densify_grad_threshold,
                        self.hparams.density_min_threshold,
                        self.hparams.max_screen_size,
                        self.hparams.max_scale,
                        self.hparams.max_num_gaussians,
                        self.hparams.densify_scale_threshold,
                        self.bbox
                    )
                # if self.global_step % self.hparams.opacity_reset_interval == 0:
                #     self.gaussians.reset_density()

            # Prune nan
            prune_mask = torch.isnan(self.gaussians.get_density).squeeze()
            if prune_mask.sum() > 0:
                self.gaussians.prune_points(prune_mask)
            
            prune_mask = (torch.isnan(self.gaussians.get_density)).squeeze()
            if prune_mask.sum() > 0:
                self.gaussians.prune_points(prune_mask)

    def on_validation_start(self):
        # torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):


        logs = {}
        return logs

    def validation_epoch_end(self, outputs):
        os.makedirs(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}', exist_ok=True)
        torch.save(
            (self.gaussians.capture(), self.current_epoch),
            f'ckpts/{hparams.dataset_name}/{hparams.exp_name}' + "/chkpnt" + str(self.current_epoch) + ".pth",
        )
        #self.gaussians.save_ply(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}' + "point_cloud.ply")
        self.scene.save(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',self.global_step,query,self.tv_vol_center,
                    self.tv_vol_nVoxel,
                    self.tv_vol_sVoxel,self.hparams)
        print(self.gaussians.get_xyz.shape[0])
        

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()   
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = GSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0,
                      precision=32,
                      log_every_n_steps=1)
    trainer.fit(system)