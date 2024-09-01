"""Script for generating degraded frames for temporary testing."""
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

import cv2
import numpy as np
from tqdm import tqdm

from VP_code.data.Data_Degradation.util import degradation_video_list_4

if __name__ == '__main__':
    root_dir = '/scratch/linshan/datasets/DAVIS'
    input_dir = os.path.join(root_dir, 'DAVIS_GT')

    # Random choose 10 clips only for temporary testing
    clip_dirs = random.sample((os.listdir(input_dir)), 10)
    for clip_dir in tqdm(clip_dirs):
        gt_dir = os.path.join(root_dir, 'DAVIS_sub_GT', clip_dir)
        degraded_dir = os.path.join(root_dir, 'DAVIS_sub_degraded', clip_dir)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(degraded_dir, exist_ok=True)

        clip_path = os.path.join(input_dir, clip_dir)
        frame_filenames = os.listdir(clip_path)
        if len(frame_filenames) > 60:
            frame_filenames = frame_filenames[:60]

        gt_imgs = []
        for frame_filename in frame_filenames:
            if frame_filename.startswith('.') or not frame_filename.endswith('jpg'):
                continue
            gt_imgs.append(cv2.imread(os.path.join(clip_path, frame_filename)) / 255.)

        degraded_imgs, gt_imgs = degradation_video_list_4(gt_imgs, '/scratch/linshan/datasets/noise_data')

        for i, frame_filename in enumerate(frame_filenames):
            if frame_filename.startswith('.') or not frame_filename.endswith('jpg'):
                continue
            output_filename = os.path.splitext(frame_filename)[0] + '.png'
            cv2.imwrite(os.path.join(gt_dir, output_filename), (np.clip(gt_imgs[i], 0, 1) * 255.).astype(np.uint8))
            cv2.imwrite(os.path.join(degraded_dir, output_filename), (np.clip(degraded_imgs[i], 0, 1) * 255.).astype(np.uint8))
