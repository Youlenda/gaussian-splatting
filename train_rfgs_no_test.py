
import torch
from utils.tensorf_utils import N_to_reso, cal_n_samples
from renderer_rf import *
from dataLoader import dataset_dict

from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams, PipelineParams
import uuid
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_iterations = 15000

# ### TensoRF ###
class SimpleSampler:
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

# args
datadir = './data/nerf_synthetic/lego'

batch_size = 4096

N_voxel_init = 2097156
N_voxel_final = 27000000
upsamp_list = [2000, 3000, 4000, 5500, 7000]
update_AlphaMask_list = [2000, 4000]

N_vis = 5
vis_every = 5000

n_lamb_sigma = [16, 16, 16]
n_lamb_sh = [48, 48, 48]
model_name = 'TensorVMSplit'

L1_weight_inital = 8e-5
L1_weight_rest = 4e-5

nSamples = 1e6
step_ratio = 0.5
alpha_mask_thre = 0.0001
distance_scale = 25
density_shift = -10

lr_init = 0.02
lr_basis = 1e-3
lr_decay_target_ratio = 0.1

# dataset
dataset_rf = dataset_dict['blender'](datadir)

# log file
logfolder_rf = './log/tensorf'
summary_writer_rf = SummaryWriter(logfolder_rf)

# parameters
aabb = dataset_rf.scene_bbox.to(device)
reso_cur = N_to_reso(N_voxel_init, aabb)
nSamples = min(nSamples, cal_n_samples(reso_cur, step_ratio)) # 1e6 | 443

# model
tensorf = eval(model_name)(aabb, reso_cur, device, density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                           near_far=dataset_rf.near_far, alphaMask_thres=alpha_mask_thre, density_shift=density_shift,
                           distance_scale=distance_scale, step_ratio=step_ratio)

# Combine train and test datasets for TensoRF
allrays_rf, allrgbs_rf = dataset_rf.all_rays, dataset_rf.all_rgbs
allrays_rf, allrgbs_rf = tensorf.filtering_rays(allrays_rf, allrgbs_rf, bbox_only=True)
Sampler_rf = SimpleSampler(allrays_rf.shape[0], batch_size)

# optimizer 
grad_vars_rf = tensorf.get_optparam_groups(lr_init, lr_basis)
optimizer_rf = torch.optim.Adam(grad_vars_rf, betas=(0.9,0.99))
lr_factor_rf = (lr_decay_target_ratio) ** (1/n_iterations)
L1_reg_weight = L1_weight_inital

# upsample
N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(N_voxel_init), np.log(N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

### Gaussion Splatting ###
def prepare_output_and_logger(model_path):   
    # if no model path get consider one
    if not model_path: # 
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else: #
            unique_str = str(uuid.uuid4())
        model_path = os.path.join("./output/", unique_str[0:10]) # ./output/8c0b6b47-c

    # Set up output folder
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(model_path)
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# parser
parser = ArgumentParser(description="Training script parameters")

lp = ModelParams(parser)
op = OptimizationParams(parser)
pp = PipelineParams(parser)
args = parser.parse_args(sys.argv[1:])

dataset_gs = lp.extract(args) # in fact, model params
opt = op.extract(args)
pipe = pp.extract(args)

# parameters
# testing_iterations = [7000, 30000]
# saving_iterations = [7000, 30000]
testing_iterations = [7000, 15000]
saving_iterations = [7000, 15000]
checkpoint_gs_iterations = []
debug_from = -1

# some other parameters
first_iter = 0
tb_writer = prepare_output_and_logger(model_path='')
sh_degree = 3
gaussians = GaussianModel(sh_degree) # what does it return?
scene = Scene(dataset_gs, gaussians)
gaussians.training_setup(opt)

bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device=device)

iter_start = torch.cuda.Event(enable_timing=True)
iter_end = torch.cuda.Event(enable_timing=True)

ema_loss_for_log = 0.0 # for exponentially moving avg?

#### TRAINING ... ####
# rf
torch.cuda.empty_cache()
PSNRs_rf = []

# gs
progress_bar = tqdm(range(first_iter, n_iterations), desc="Training progress")
first_iter += 1

for iteration in range(first_iter, n_iterations + 1):
    # TensoRF
    ray_idx = Sampler_rf.next_ids()
    ray_rf, rgb_gt_rf = allrays_rf[ray_idx], allrgbs_rf[ray_idx].to(device)

    rgb_map_rf, depth_map_rf = OctreeRender_trilinear_fast(ray_rf, tensorf, chunk=batch_size, N_samples=nSamples, is_train=True, device=device)

    # loss
    loss_rf = torch.mean((rgb_map_rf - rgb_gt_rf) ** 2) 
    total_loss_rf = loss_rf
    loss_reg_L1_rf = tensorf.density_L1()
    total_loss_rf += L1_reg_weight * loss_reg_L1_rf
    summary_writer_rf.add_scalar('reg_l1', loss_reg_L1_rf.detach().item(), global_step=iteration)

    optimizer_rf.zero_grad()
    total_loss_rf.backward()
    optimizer_rf.step()

    loss_rf = loss_rf.detach().item()

    PSNRs_rf.append(-10.0 * np.log(loss_rf) / np.log(10.0))
    summary_writer_rf.add_scalar('PSNR', PSNRs_rf[-1], global_step=iteration)
    summary_writer_rf.add_scalar('mse', loss_rf, global_step=iteration)

    for param_group_rf in optimizer_rf.param_groups:
        param_group_rf['lr'] = param_group_rf['lr'] * lr_factor_rf

    if iteration in update_AlphaMask_list:
        if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3:
            reso_mask = reso_cur
        new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
        if iteration == update_AlphaMask_list[0]:
            tensorf.shrink(new_aabb)
            L1_reg_weight = 4e-5
        if iteration == update_AlphaMask_list[1]:
            allrays_rf, allrgbs_rf = tensorf.filtering_rays(allrays_rf, allrgbs_rf)
            Sampler_rf = SimpleSampler(allrays_rf.shape[0], batch_size)
    
    if iteration in upsamp_list:
        n_voxels = N_voxel_list.pop(0)
        reso_cur = N_to_reso(n_voxels, tensorf.aabb)
        nSamples = min(1e6, cal_n_samples(reso_cur, step_ratio))
        tensorf.upsample_volume_grid(reso_cur)
        grad_vars_rf = tensorf.get_optparam_groups(lr_init, lr_basis)
        optimizer_rf = torch.optim.Adam(grad_vars_rf, betas=(0.9, 0.99))
    
###############################################################################################################################################################################################
    # Gaussion Splatting
    if network_gui.conn == None:
        network_gui.try_connect()
    iter_start.record()
    gaussians.update_learning_rate(iteration)

    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()

    # Pick a random Camera
    viewpoint_stack = scene.getTrainCameras().copy() 
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

    # Render
    bg = torch.rand((3), device=device) if opt.random_background else background

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image, viewspace_point_tensor, visibility_filter, radii, depth_gs = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

    # Loss image
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    # Depth Loss
    # print('depth_map_rf', depth_map_rf.shape)
    # print('depth_gs', depth_gs.shape)
    # assert False

    loss.backward()

    iter_end.record()

    with torch.no_grad():
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_description(
            f'Gaussian Splatting - loss: {ema_loss_for_log:.4f}; '
            f'TensoRF - psnr: {float(np.mean(PSNRs_rf)):.2f}')
            progress_bar.update(10)
        if iteration == n_iterations:
            progress_bar.close()

        # Log and save
        training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
        if (iteration in saving_iterations):
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        # Densification
        if iteration < opt.densify_until_iter:
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter]) # size: 95122
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            
            if iteration % opt.opacity_reset_interval == 0 or (dataset_gs.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

        # Optimizer step
        if iteration < n_iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if (iteration in checkpoint_gs_iterations):
            print("\n[ITER {}] Saving GS Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

tensorf.save('./log/tensorf_model.th')
