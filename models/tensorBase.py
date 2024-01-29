import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time


def positional_encoding(positions, n_freqs):
    freqs = torch.pow(2.0, torch.arange(n_freqs)).to(positions.device)
    scaled_positions = positions.unsqueeze(-1) * freqs # (877, 3, 2)
    scaled_positions = scaled_positions.view(positions.shape[0], -1) # (877, 6)
    encoded_positions = torch.cat([torch.sin(scaled_positions), torch.cos(scaled_positions)], dim=-1) # (877, 12)
    return encoded_positions

def raw2alpha(sigma, dist): # faster than mine
    alpha = 1.0 - torch.exp(-sigma * dist)
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
    weights = alpha * T[:, :-1]
    return weights

class AlphaGridMask(torch.nn.Module):
    def __init__(self, aabb, alpha_volume, device):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1
    
    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1) # 244908
        return alpha_vals

class MLPRender(torch.nn.Module):
    def __init__(self):
        super(MLPRender, self).__init__()
        in_channel = 3 + 27 + 3 * 4 + 27 * 4
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_channel, 128),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(128, 128),
                                       torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(128, 3))
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, view_dirs, features): # (877, 3), (877, 27)
        in_data = [view_dirs, features] # a list of 2 things
        in_data += [positional_encoding(view_dirs, 2)] # add to list
        in_data += [positional_encoding(features, 2)] # add to list
        mlp_input = torch.cat(in_data, dim=-1) # merge the list -> (877, 150)
        rgb = torch.sigmoid(self.mlp(mlp_input)) # (877, 3)
        return rgb

class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp=[16,16,16], appearance_n_comp=[48,48,48], alphaMask=None, near_far=[2.0,6.0],
                    density_shift=-10, alphaMask_thres=0.0001, distance_scale=25, rayMarch_weight_thres=0.0001, step_ratio=0.5):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]

        self.init_svd_volume(device)

        self.renderModule = MLPRender().to(device)

    def update_stepSize(self, gridSize):
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag/self.stepSize).item()) + 1

    def init_svd_volume(self, device):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass

    def feature2density(self, density_features):
        return F.softplus(density_features + self.density_shift)
    
    def shrink(self, new_aabb, voxel_size):
        pass
    
    def normalize_coord(self, xyz_sampled):
        # return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1
        return (xyz_sampled - self.aabb[0]) / (self.aabbSize/2.0) - 1 

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device), self.device,)
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('filtering rays ...')
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(torch.linspace(0, 1, gridSize[0]),
                                             torch.linspace(0, 1, gridSize[1]),
                                             torch.linspace(0, 1, gridSize[2])), -1).to(self.device)
        # convert samples to be within a bbox
        dense_xyz = (1.0 - samples) * self.aabb[0] + samples * self.aabb[1]

        alpha = torch.zeros(gridSize[0], gridSize[1], gridSize[2]).to(self.device)
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(128,128,128)):
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None, None]
        dense_xyz = dense_xyz.transpose(0,2).contiguous()

        # To make the volume slightly dilated, ensuring that even slightly occupied voxels get considered.
        alpha = F.max_pool3d(alpha, kernel_size=3, padding=1, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.aabb, alpha, self.device)

        valid_xyz = dense_xyz[alpha>0.5]
        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)
        new_aabb = torch.stack((xyz_min, xyz_max))

        return new_aabb

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1.0 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        return alpha


    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        viewdirs = rays_chunk[:, 3:6]
        xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        weight = raw2alpha(sigma, dists * self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map

