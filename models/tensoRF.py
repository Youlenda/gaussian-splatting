

from .tensorBase import *

class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0])))) # why 1 then 0?
            line_coef.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    def init_svd_volume(self, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), 27, bias=False).to(device)

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_netwprk=0.001):
        grad_vars = [{'params': self.density_line,  'lr':lr_init_spatial},
                     {'params': self.density_plane, 'lr':lr_init_spatial},
                     {'params': self.app_line,      'lr':lr_init_spatial},
                     {'params': self.app_plane,     'lr':lr_init_spatial},
                     {'params': self.renderModule.parameters(), 'lr': lr_init_netwprk},
                     {'params': self.basis_mat.parameters(),    'lr': lr_init_netwprk}
                     ]
        return grad_vars

    def density_L1(self):
        total = 0
        for i in range(len(self.density_plane)):
            total += torch.mean(torch.abs(self.density_plane[i])) + torch.mean(torch.abs(self.density_line[i])) # eq. 18
        return total
    
    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        # F.grid_sample: sample data at non-integer coordinates by trilinear interpolation
        for idx in range(len(self.density_plane)): # 3
            plane_coef_point = F.grid_sample(self.density_plane[idx], coordinate_plane[[idx]], align_corners=True).view(-1, *xyz_sampled.shape[:1]) # (1, 16, 128, 128), (1, 985211, 1, 2) -> (1, 16, 985211, 1) -> (16, 985211)
            line_coef_point = F.grid_sample(self.density_line[idx], coordinate_line[[idx]], align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature += torch.sum(plane_coef_point * line_coef_point, dim=0) # (985211)
        
        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2) # (3, 2, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx in range(len(self.app_plane)): # 3
            plane_coef_point.append(F.grid_sample(self.app_plane[idx], coordinate_plane[[idx]], align_corners=True).view(-1, *xyz_sampled.shape[:1])) # (1, 48, 128, 128), (1, 2, 1, 2) -> (1, 2, 48, 1) -> (48, 2) 
            line_coef_point.append(F.grid_sample(self.app_line[idx], coordinate_line[[idx]], align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point) # (144, 2), (144, 2)
        
        return self.basis_mat((plane_coef_point*line_coef_point).T) # (2, 144) -> (2, 27)

    @torch.no_grad()
    def upsample_volume_grid(self, target_res):
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, target_res)
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, target_res)

        self.update_stepSize(target_res)
        print(f'Upsampling to {target_res}')

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, target_res):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(F.interpolate(plane_coef[i].data, size=(target_res[mat_id_1],target_res[mat_id_0]), mode='bilinear', align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(line_coef[i].data, size=(target_res[vec_id],1), mode='bilinear', align_corners=True))
        return plane_coef, line_coef

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:])
            self.app_line[i] = torch.nn.Parameter(self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:])
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]])
            self.app_plane[i] = torch.nn.Parameter(self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]])

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))