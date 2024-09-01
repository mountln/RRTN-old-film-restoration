import argparse
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from VP_code.models.RAFT_core.raft import RAFT


def get_raft(model_path="pretrained_models/raft-sintel.pth"):
    opts = argparse.Namespace()
    opts.model = model_path
    opts.dataset = None
    opts.small = False
    opts.mixed_precision = False
    opts.alternate_corr = False

    model = torch.nn.DataParallel(RAFT(opts))
    model.load_state_dict(torch.load(opts.model))

    model = model.module
    model.eval()

    return model


def check_flow_occlusion(flow_f, flow_b):
    """
    Compute occlusion map through forward/backward flow consistency check
    """
    def get_occlusion(flow1, flow2):
        grid_flow = grid + flow1
        grid_flow[0, :, :] = 2.0 * grid_flow[0, :, :] / max(W - 1, 1) - 1.0
        grid_flow[1, :, :] = 2.0 * grid_flow[1, :, :] / max(H - 1, 1) - 1.0
        grid_flow = grid_flow.permute(1, 2, 0)
        flow2_inter = torch.nn.functional.grid_sample(flow2[None, ...], grid_flow[None, ...])[0]
        score = torch.exp(- torch.sum((flow1 + flow2_inter) ** 2, dim=0) / 2.)
        occlusion = (score > 0.5)
        return occlusion[None, ...].float()

    C, H, W = flow_f.size()
    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W)
    yy = yy.view(1, H, W)
    grid = torch.cat((xx, yy), 0).float().cuda()

    occlusion_f = get_occlusion(flow_f, flow_b)
    occlusion_b = get_occlusion(flow_b, flow_f)
    flow_f = torch.cat((flow_f, occlusion_f), 0)
    flow_b = torch.cat((flow_b, occlusion_b), 0)

    return flow_f, flow_b
