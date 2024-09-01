import sys
import os
import cv2
import importlib
import argparse
import yaml

sys.path.append(os.path.dirname(sys.path[0]))

import torch

from VP_code.data.dataset import Film_dataset_1
from VP_code.utils.util import frame_to_video
from VP_code.utils.data_util import tensor2img
from VP_code.metrics.psnr_ssim import calculate_psnr

from torch.utils.data import DataLoader


def load_model(opts, which_model='first'):
    assert which_model in ['first', 'second']

    net = importlib.import_module('VP_code.models.' + opts.model_name)
    netG = net.Video_Backbone()

    if which_model == 'first':
        model_path = opts.model_path_first
    elif which_model == 'second':
        model_path = opts.model_path_second
    else:
        raise ValueError('`which_model` should be "first" or "second"')

    checkpoint = torch.load(model_path)
    netG.load_state_dict(checkpoint['netG'])
    netG.cuda()
    print("Finish loading model ...")

    return netG


def load_dataset(config_dict):

    val_dataset = Film_dataset_1(config_dict['datasets']['val'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, sampler=None)
    print("Finish loading dataset ...")
    print("Test set statistics:")
    print(f'\n\tNumber of test videos: {len(val_dataset)}')

    return val_loader


def validation(opts, config_dict, loaded_model, val_loader, recursion_step=1):

    psnr = 0.0
    loaded_model.eval()

    for val_data in val_loader:

        val_frame_num = config_dict['val']['val_frame_num']
        all_len = val_data['lq'].shape[1]
        all_output = []

        clip_name, _ = val_data['key'][0].split('/')
        test_clip_par_folder = val_data['video_name'][0]

        frame_name_list = val_data['name_list']

        part_output = None
        for i in range(0, all_len, opts.temporal_stride):
            current_part = {}
            current_part['lq'] = val_data['lq'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['gt'] = val_data['gt'][:, i:min(i + val_frame_num, all_len), :, :, :]
            current_part['key'] = val_data['key']
            current_part['frame_list'] = val_data['frame_list'][i:min(i + val_frame_num, all_len)]

            part_lq = current_part['lq'].cuda()

            with torch.no_grad():
                try:
                    part_output = loaded_model(part_lq)
                except RuntimeError as e:
                    print("Warning: runtime error", e)
                    part_output = part_lq.clone()

            if i == 0:
                all_output.append(part_output.detach().cpu().squeeze(0))
            else:
                restored_temporal_length = min(i + val_frame_num, all_len) - i - (
                    val_frame_num - opts.temporal_stride)
                all_output.append(part_output[:, 0 - restored_temporal_length:, :, :, :]
                                  .detach().cpu().squeeze(0))

            del part_lq

            if (i + val_frame_num) >= all_len:
                break

        val_output = torch.cat(all_output, dim=0)
        gt = val_data['gt'].squeeze(0)
        lq = val_data['lq'].squeeze(0)
        if config_dict['datasets']['val']['normalizing']:
            val_output = (val_output + 1) / 2
            gt = (gt + 1) / 2
            lq = (lq + 1) / 2
        torch.cuda.empty_cache()

        gt_imgs = []
        sr_imgs = []
        for j in range(len(val_output)):
            gt_imgs.append(tensor2img(gt[j]))
            sr_imgs.append(tensor2img(val_output[j]))

        # Save the image
        for id, sr_img in enumerate(sr_imgs):
            save_place = os.path.join(opts.save_place, opts.name, 'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step), test_clip_par_folder, clip_name, frame_name_list[id][0])
            dir_name = os.path.abspath(os.path.dirname(save_place))
            os.makedirs(dir_name, exist_ok=True)
            cv2.imwrite(save_place, sr_img)

        if test_clip_par_folder == os.path.basename(opts.input_video_url):
            input_clip_url = os.path.join(opts.input_video_url, clip_name)
        else:
            input_clip_url = os.path.join(opts.input_video_url, test_clip_par_folder, clip_name)

        restored_clip_url = os.path.join(opts.save_place, opts.name, 'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step), test_clip_par_folder, clip_name)
        video_save_url = os.path.join(opts.save_place, opts.name, 'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step), test_clip_par_folder, clip_name + '.avi')
        frame_to_video(input_clip_url, restored_clip_url, video_save_url)

        psnr_this_video = [calculate_psnr(sr, gt) for sr, gt in zip(sr_imgs, gt_imgs)]
        psnr += sum(psnr_this_video) / len(psnr_this_video)

    psnr /= len(val_loader)

    print(f'# PSNR between input and output: {psnr}')
    return psnr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='The name of this experiment')
    parser.add_argument('--model_name', type=str, default='', help='The name of adopted model')
    parser.add_argument('--model_path_first', type=str, default='pretrained_models/rrtn_128_first.pth', help='Path to the first pretrained model')
    parser.add_argument('--model_path_second', type=str, default='pretrained_models/rrtn_128_second.pth', help='Path to the second pretrained model')
    parser.add_argument('--input_video_url', type=str, default='', help='degraded video input')
    parser.add_argument('--temporal_length', type=int, default=15, help='How many frames should be processed in one forward')
    parser.add_argument('--temporal_stride', type=int, default=3, help='Stride value while sliding window')
    parser.add_argument('--recursion_threshold', type=float, default=43, help='Threshold for further recursion. If PSNR between the input video and output video is smaller than the threshold, the recursion will be continued. Default is 43.')
    parser.add_argument('--max_recursion', type=int, default=4, help='Max recursion steps. Default is 4.')
    parser.add_argument('--save_place', type=str, default='OUTPUT', help='save place')

    opts = parser.parse_args()

    with open(os.path.join('./configs', opts.name + '.yaml'), 'r') as stream:
        config_dict = yaml.safe_load(stream)

    # ===================
    # The first recursion
    config_dict['datasets']['val']['dataroot_gt'] = opts.input_video_url
    config_dict['datasets']['val']['dataroot_lq'] = opts.input_video_url
    config_dict['val']['val_frame_num'] = opts.temporal_length

    loaded_model = load_model(opts, which_model='first')
    val_loader = load_dataset(config_dict)

    print('=======')
    print('Recursion: 1')
    psnr = validation(opts, config_dict, loaded_model, val_loader, recursion_step=1)

    # ===================
    # Further recursions
    loaded_model = load_model(opts, which_model='second')
    for i in range(opts.max_recursion - 1):
        if psnr >= opts.recursion_threshold:
            break

        recursion_step = i + 2
        print('=======')
        print(f'Recursion: {recursion_step}')

        video_url = os.path.join(
            opts.save_place,
            opts.name,
            'test_results_' + str(opts.temporal_length) + "_rec" + str(recursion_step - 1)
        )
        config_dict['datasets']['val']['dataroot_gt'] = video_url
        config_dict['datasets']['val']['dataroot_lq'] = video_url

        val_loader = load_dataset(config_dict)

        psnr = validation(opts, config_dict, loaded_model, val_loader, recursion_step=recursion_step)
