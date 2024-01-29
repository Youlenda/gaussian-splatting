
import json
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from kornia import create_meshgrid
from tqdm import tqdm


class BlenderDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = datadir

        self.transform = T.ToTensor()
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.white_bg = True
        self.near_far = [2.0, 6.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.img_wh = (800,800)

        self.read_meta()

    def read_meta(self):
        self.all_rgbs = []
        self.all_rays = []
        self.all_imgs = []

        # Calculate the total number of frames beforehand
        total_frames = sum([len(json.load(open(f'{self.datadir}/transforms_{split}.json', 'r'))['frames']) for split in ['train', 'test']])
        pbar = tqdm(total=total_frames, desc='Loading data')

        for split in ['train', 'test']:
            with open(f'{self.datadir}/transforms_{split}.json', 'r') as file:
                meta = json.load(file)
            
            focal = (800/2.0) / np.tan(meta['camera_angle_x']/2.0)
            directions = get_ray_direction(800, 800, focal)
            directions = directions/torch.norm(directions, dim=-1, keepdim=True)

            for frame in meta['frames']:
                pbar.update(1)
                img = Image.open(f'{self.datadir}/{frame["file_path"]}.png')
                img = self.transform(img) # (4, 800, 800)

                img = img.view(4, -1).permute(1, 0) # (640000, 4)
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:]) # RGB * A + RGB_background * (1 - A)
                self.all_imgs += [img.view(800, 800, 3).permute(2, 1, 0)] # (3, 800, 800)
                self.all_rgbs += [img]

                pose = np.array(frame['transform_matrix']) @ self.blender2opencv  # transform matrix * blender2opencv = pose = c2w
                c2w = torch.FloatTensor(pose)
                rays_o, rays_d = get_rays(directions, c2w)
                self.all_rays += [torch.cat([rays_o, rays_d], -1)]

        pbar.close()

        self.all_rgbs = torch.cat(self.all_rgbs, 0)   # [192000000, 3]
        self.all_rays = torch.cat(self.all_rays, 0)
        self.all_imgs = torch.stack(self.all_imgs, 0) # [300, 3, 800, 800]

    # def __getitem__(self, idx):

    #     if self.split == 'train':  # use data in the buffers
    #         sample = {'rays': self.all_rays[idx],
    #                   'rgbs': self.all_rgbs[idx]}

    #     else:  # create data for each image separately

    #         img = self.all_rgbs[idx]
    #         rays = self.all_rays[idx]
    #         mask = self.all_masks[idx] # for quantity evaluation

    #         sample = {'rays': rays,
    #                   'rgbs': img,
    #                   'mask': mask}
    #     return sample


def get_ray_direction(w, h, focal):
    grid = create_meshgrid(w, h, normalized_coordinates=False)[0] + 0.5 # (1, 800, 800, 2) -> (800, 800, 2)
    i, j = grid.unbind(-1) # (800, 800), (800, 800)
    # Normalize x and y axis, add z axis
    directions = torch.stack([(i - 800*0.5)/focal, (j - 800*0.5)/focal, torch.ones_like(i)], -1) # (800, 800, 3)
    return directions

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T      # (800, 800, 3) # ? why transpose?
    rays_o = c2w[:3, 3].expand(rays_d.shape) # (800, 800, 3)

    rays_d = rays_d.view(-1, 3) # (640000, 3)
    rays_o = rays_o.view(-1, 3) # (640000, 3)
    return rays_o, rays_d
