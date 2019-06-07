"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.utils import get_paddings_indicator
from torchplus.nn import Empty
from torchplus.tools import change_default_args
from .pointpillars import PFNLayer


class PFNLayerForCudaImplementation(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer. Add batch size to fit tensorrt
        Modified batchnorm 1d -> 2d to fit tensorrt
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """
        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:
            # Have to use BatchNorm2d since 1d, 1, 1200000, 64 , 1200000is too long
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            # Linear = change_default_args(bias=False)(nn.Linear)
            Conv = change_default_args(bias=False)(nn.Conv2d)
        else:
            BatchNorm1d = Empty
            # Linear = change_default_args(bias=True)(nn.Linear)
            Conv = change_default_args(bias=False)(nn.Conv2d)

        self.norm = BatchNorm1d(self.units)
        self.conv = Conv(in_channels, self.units, 1)
        # replace max
        self.dilaconv = Conv(self.units, self.units, (1, 34), dilation=(1, 3))

    def forward(self, inputs):
        # ############# the conversion is obeying the rule:
        ############### tensorrt is not supporting much of indexing, broadcasting, and wired batching
        # the following struct
        # (1, 12000, 100, 4)
        x = self.conv(inputs)
        x = self.norm(x)
        x = F.relu(x)
        # replace max to dilated conv
        # x_max = torch.max(x, dim=1, keepdim=True)[0]

        x_max = self.dilaconv(x)
        if self.last_vfe:
            return x_max


class PillarFeatureNetForCudaImplementation(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Modified Pillar Feature Net to match the c++ codes.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayerForCudaImplementation(in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self,
                dev_pillar_x_, dev_pillar_y_, dev_pillar_z_, dev_pillar_i_,
                dev_num_points_per_pillar_, dev_x_coors_for_sub_shaped_, dev_y_coors_for_sub_shaped_,
                dev_pillar_feature_mask_):
        """
        The c++ code inputs
        :param dev_pillar_x_: (2, 1, 12000, 100)
        :param dev_pillar_y_: (2, 1, 12000, 100)
        :param dev_pillar_z_: (2, 1, 12000, 100)
        :param dev_pillar_i_: (2, 1, 12000, 100)
        :param dev_num_points_per_pillar_: num_voxels in pytorch code, (2, 1, 12000), the only one without 100
        :param dev_x_coors_for_sub_shaped_: (2, 1, 12000, 100)
        :param dev_y_coors_for_sub_shaped_: (2, 1, 12000, 100)
        :param dev_pillar_feature_mask_: (2, 1, 12000, 100)  point wise mask
        :return:
        """
        # train reshape, onnx conersion not necessary
        dev_num_points_per_pillar_= torch.reshape(dev_num_points_per_pillar_, (dev_pillar_x_.shape[0], 1, 12000, 1))

        xyz = torch.cat([dev_pillar_x_, dev_pillar_y_, dev_pillar_z_], dim=1)
        xyz_mean = xyz.sum(dim=3, keepdim=True) / dev_num_points_per_pillar_
        xyz_submean = xyz - xyz_mean
        '''
        # check if have nan
        def check_is_nan(input):
            is_nan = torch.isnan(input)
            print(torch.sum(torch.isnan(input)))
            print(input[is_nan][0])
            exit()
        check_is_nan(x_submean)
        '''
        # ############ work above ###############################
        # trt is not supporting broadcasting sometimes
        vx = torch.tensor(self.vx, device="cuda", dtype=torch.float).view(1, 1, 1, 1)
        vy = torch.tensor(self.vy, device="cuda", dtype=torch.float).view(1, 1, 1, 1)
        x_offset = torch.tensor(self.x_offset, device="cuda", dtype=torch.float).view(1, 1, 1, 1)
        y_offset = torch.tensor(self.y_offset, device="cuda", dtype=torch.float).view(1, 1, 1, 1)
        x_center = dev_x_coors_for_sub_shaped_ * vx + x_offset
        y_center = dev_y_coors_for_sub_shaped_ * vy + y_offset

        # Find distance of x, y, and z from pillar center
        x_subcenter = dev_pillar_x_ - x_center
        y_subcenter = dev_pillar_y_ - y_center
        features_ls = [xyz, dev_pillar_i_, xyz_submean,
                       x_subcenter, y_subcenter]

        features = torch.cat(features_ls, dim=1)
        features_masked = features.mul(dev_pillar_feature_mask_)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features_out = pfn(features_masked)
        # return features_out.squeeze() can not simply squeeze because the batch sometimes is 1!!
        return torch.squeeze(features_out, 3)


class PointPillarsScatterForCudaImpmentation(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        """
        Point Pillar's Scatter.
        add batch dimension
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size):
        # now the voxel_features is (2, 12000, 64) instead of (xxx, 64)
        # batch_canvas will be the final output.
        # [2, 64, 12000] -> [2, 12000, 64]
        voxel_features = voxel_features.permute(0, 2, 1).contiguous()
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            non_empty_pillar_num = this_coords.shape[
                0]  # one error: orginal pfe final squeeze is not considering batchsize==1
            # print("nonempty pillar num vs feature dims: ", non_empty_pillar_num, voxel_features.shape)
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            # take out the batch
            voxels = voxel_features[batch_itt, :non_empty_pillar_num, :]
            voxels = voxels.t()
            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels
            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        return batch_canvas
