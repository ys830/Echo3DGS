from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1
        return len(self.poses)

    def __getitem__(self, idx):
        # if self.split.startswith('train'):
        #     img_idxs = np.random.choice(len(self.poses), self.batch_size)
        #     pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
        #     rays = self.rays[img_idxs, pix_idxs]
        #     sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
        #               'rgb': rays[:, :3]}
        # else:
        #     sample = {'pose': self.poses[idx], 'img_idxs': idx}
        #     if len(self.rays)>0: # if ground truth available
        #         rays = self.rays[idx]
        #         sample['rgb'] = rays[:, :3]

        # return sample
        #sample = {'pose': self.poses, 'img': self.rays}
        return 1