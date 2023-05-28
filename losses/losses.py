import torch
import torch.nn as nn
from einops import rearrange


# class DownMSELoss(nn.Module):
#     def __init__(self, size=8):
#         super().__init__()
#         self.avgpooling = nn.AvgPool2d(kernel_size=size)
#         self.tot = size * size
#         self.mse = nn.MSELoss(reduction='sum')

#     def forward(self, dmap, gt_density):
#         gt_density = self.avgpooling(gt_density) * self.tot
#         b, c, h, w = dmap.size()
#         assert gt_density.size() == dmap.size()
#         return self.mse(dmap, gt_density)

# class DownMSELoss(nn.Module):
#     def __init__(self, size=8):
#         super().__init__()
#         self.avgpooling = nn.AvgPool2d(kernel_size=size)
#         self.tot = size * size
#         self.mse = nn.MSELoss(reduction='sum')

#     def forward(self, dmap, gt_density):
#         gt_density = self.avgpooling(gt_density) * self.tot
#         b, c, h, w = dmap.size()
#         assert gt_density.size() == dmap.size()
#         res = dmap.clone()
#         res[gt_density==0] = 0
#         res_loss = self.mse(res, gt_density)
#         tot_loss = self.mse(dmap, gt_density)
#         return res_loss + 0.5 * tot_loss

class DownMSELoss(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=size)
        self.tot = size * size
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, dmap, gt_density):
        gt_density = self.avgpooling(gt_density) * self.tot
        b, c, h, w = dmap.size()
        assert gt_density.size() == dmap.size()
        res = dmap.clone()
        not_res = dmap.clone()
        res[gt_density==0] = 0
        res_loss = self.mse(res, gt_density)
        not_res[gt_density>0] = gt_density[gt_density>0]
        not_res_loss = self.mse(not_res, gt_density)
        return res_loss + 0.8 * not_res_loss