
import os
from tqdm.auto import tqdm
from opt import config_parser

import json, random
from renderer_rf import *
from utils.tensorf_utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

class SimpleSampler: # mine
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.current = 0
        self.shuffle_ids()
    def shuffle_ids(self):
        self.ids = torch.randperm(self.total)
    def next_ids(self):
        if self.current + self.batch > self.total:
            self.shuffle_ids()
            self.current = 0
        batch_ids = self.ids[self.current:self.current+self.batch]
        self.current += self.batch
        return batch_ids

def N_to_reso(n_voxel, aabb): # mine
    aabb_size = aabb[1] - aabb[0]
    voxel_size = (aabb_size.prod()/n_voxel)**(1/3)
    return (aabb_size / voxel_size).long().tolist()

def cal_n_samples(reso, step_size=0.5): # mine
    return ((torch.sqrt(torch.sum(torch.square(torch.tensor(reso)))) / step_size).long()).tolist()

def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', is_stack=False)
    test_dataset = dataset(args.datadir, split='test', is_stack=True)

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    
    # init log file
    logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis',  exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba',      exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio)) # 1e6 | 443

    tensorf = eval(args.model_name)(aabb, reso_cur, device, density_n_comp=args.n_lamb_sigma, appearance_n_comp=args.n_lamb_sh,
                                    near_far=train_dataset.near_far, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift,
                                    distance_scale=args.distance_scale, step_ratio=args.step_ratio)
    
    # init optimizer 
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis) # 0.02, 0.001
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    lr_factor = args.lr_decay_target_ratio ** (1/args.n_iters) # (0.1)**(1/15000)=0.9998
    
    # Number of voxel for each upsample list
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(args.upsamp_list)+1))).long()).tolist()[1:] #[2097155,  3496047,  5828059,  9715614, 16196343, 26999994]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True) # 64000000 -> 63845405
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    L1_reg_weight = args.L1_weight_inital # 8e-5 different from paper

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx = trainingSampler.next_ids()
        ray_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        
        rgb_map, _ = renderer(ray_train, tensorf, chunk=args.batch_size, N_samples=nSamples, is_train=True, device=device)

        loss = torch.mean((rgb_map - rgb_train) ** 2) 
        total_loss = loss
        loss_reg_L1 = tensorf.density_L1()
        total_loss += L1_reg_weight * loss_reg_L1
        summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0)) # just rgb loss (mse)
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor # 0.02*0.9998=0.0199969

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset, tensorf, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=nSamples)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                L1_reg_weight = args.L1_weight_rest

            if iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            print("reset lr to initial")
            grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
    tensorf.save(f'{logfolder}/{args.expname}.th')

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', is_stack=True)

    # Load checkpoint
    logfolder = './log'
    ckpt = torch.load(f'{logfolder}/tensorf_lego_VM/tensorf_lego_VM.th')
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # init model and load weights
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    PSNRs_test = evaluation(test_dataset, tensorf, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'render_', N_samples=-1)

    print(f'=================> {args.expname} test all psnr: {np.mean(PSNRs_test)} <=================')

@torch.no_grad()
def export_mesh(args):
    # Load checkpoint
    logfolder = './log'
    ckpt = torch.load(f'{logfolder}/tensorf_lego_VM/tensorf_lego_VM.th')
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # init model and load weights
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    # breakpoint()
    convert_sdf_samples_to_ply(alpha.cpu(), f'mesh.ply', bbox=tensorf.aabb.cpu(), level=0.005)

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    # print(args)

    reconstruction(args)
    # render_test(args)
    # export_mesh(args)
