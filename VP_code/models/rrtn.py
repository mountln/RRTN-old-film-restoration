import torch
import torch.nn.functional as F
from torch import nn
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model.weight_init import constant_init

try:
    from VP_code.models.raft_flow import get_raft
    from VP_code.models.arch_util import ResidualBlockNoBN, flow_warp, make_layer
    from VP_code.models.Spatial_Restoration_2 import Swin_Spatial_2
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
    from VP_code.models.raft_flow import get_raft
    from VP_code.models.arch_util import ResidualBlockNoBN, flow_warp, make_layer
    from VP_code.models.Spatial_Restoration_2 import Swin_Spatial_2


class Video_Backbone(nn.Module):
    def __init__(self,
                 num_feat=16,
                 num_block=6,
                 input_channel=1,
                 input_size=128,
                 num_recursion=2,
                 max_residue_magnitude=10,
                 downscale_first=False,
                 flow_estimator_path='pretrained_models/raft-sintel.pth',
                 cpu_cache_length=100,
                 visualization=False):

        super().__init__()
        self.num_feat = num_feat
        self.num_block = num_block
        self.input_channel = input_channel
        self.num_recursion = num_recursion

        self.downscale_first = downscale_first
        self.cpu_cache_length = cpu_cache_length

        # optical flow
        self.spynet = get_raft(flow_estimator_path)

        # feature extraction module
        self.feat_extract = ConvResBlock(input_channel + 1, num_feat, 5)
        scale_factor = 1

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * num_feat,
                num_feat,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = Swin_Spatial_2(
                img_size=input_size,
                embed_dim=64,
                depths=[2, 2, 2],
                num_heads=[4, 4, 4],
                mlp_ratio=2,
                in_chans=(2 + i) * num_feat)

        # upsampling module
        self.reconstruction = ConvResBlock(
            5 * num_feat, num_feat, 5)
        self.upsample1 = PixelShufflePack(
            num_feat, num_feat, scale_factor)
        self.upsample2 = PixelShufflePack(
            num_feat, num_feat, scale_factor)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, input_channel, 3, 1, 1)
        self.img_upsample = nn.Identity()

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.visualization = visualization
        self.vis_imgs = None

    def compute_flow(self, lqs):
        n, t, c, h, w = lqs.size()
        if c == 1:
            lqs = lqs.repeat(1, 1, 3, 1, 1)
            c = 3
        forward_lqs = lqs[:, 1:, :, :, :]
        backward_lqs = lqs[:, :-1, :, :, :]

        with torch.no_grad():
            _, forward_flow = self.spynet(
                torch.cat([forward_lqs, lqs[:, :1, :, :, :]],
                          dim=1).reshape(-1, c, h, w) * 255,
                torch.cat([backward_lqs, lqs[:, 2:3, :, :, :]],
                          dim=1).reshape(-1, c, h, w) * 255,
                iters=24, test_mode=True)
            forward_flow = forward_flow.view(n, t, 2, h, w)
            _, backward_flow = self.spynet(
                torch.cat([backward_lqs, lqs[:, -1:, :, :, :]],
                          dim=1).reshape(-1, c, h, w) * 255,
                torch.cat([forward_lqs, lqs[:, -3:-2, :, :, :]],
                          dim=1).reshape(-1, c, h, w) * 255,
                iters=24, test_mode=True)
            backward_flow = backward_flow.view(n, t, 2, h, w)

        return forward_flow, backward_flow

    def propagate(self, feats, flows, module_name):
        n, t, _, h, w = flows.size()

        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.num_feat, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if not self.downscale_first:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs: torch.Tensor):
        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU (no effect if using CPU)
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        if not self.downscale_first:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # compute residual indicators
        # lqs: [n, t, c, h, w], residual indicators: [n, t, 2, h, w]
        residual_indicators = lqs.new_zeros(n, t, 2, h, w)
        for i in range(0, t):
            lq_current = lqs[:, i, :, :, :]

            # backward
            flow = flows_backward[:, i, :, :, :]
            if i != t - 1:
                warped_lq_next = flow_warp(lqs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
            else:
                warped_lq_next = flow_warp(lqs[:, -3, :, :, :], flow.permute(0, 2, 3, 1))
            residual_indicators[:, i, 0, :, :] = torch.abs(warped_lq_next[:, 0, :, :] - lq_current[:, 0, :, :])

            # forward
            if i != 0:
                flow = flows_forward[:, i - 1, :, :, :]
                warped_lq_previous = flow_warp(lqs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
            # first and last residual_indicators
            else:
                flow = flows_forward[:, -1, :, :, :]
                warped_lq_previous = flow_warp(lqs[:, 2, :, :, :], flow.permute(0, 2, 3, 1))
            residual_indicators[:, i, 1, :, :] = torch.abs(warped_lq_previous[:, 0, :, :] - lq_current[:, 0, :, :])

            # geometric mean of two residual indicators
            noise_mask = torch.sqrt(residual_indicators[:, :, :1, :, :] * residual_indicators[:, :, 1:, :, :])

        lqs_with_noise_mask = torch.cat([lqs, noise_mask], dim=2)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs_with_noise_mask[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs_with_noise_mask.view(-1, c + 1, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows[:, :-1, :, :, :], module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        """Init constant offset."""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        """Forward function."""
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


#############################
# Conv + ResBlock


class ConvResBlock(nn.Module):
    def __init__(self, in_feat, out_feat=64, num_block=30):
        super(ConvResBlock, self).__init__()

        conv_resblock = []
        conv_resblock.append(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True))
        conv_resblock.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        conv_resblock.append(make_layer(
            ResidualBlockNoBN, num_block, num_feat=out_feat))

        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_feat, out_feat, scale_factor):
        super(PixelShufflePack, self).__init__()

        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat,
                                 out_feat * scale_factor * scale_factor,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)
