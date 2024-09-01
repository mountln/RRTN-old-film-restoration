import random
import torch
from torch.utils import data as data
import os
import numpy as np
import cv2
import operator
from PIL import Image

import torchvision.transforms as transforms
try:
    from VP_code.utils.util import get_root_logger
    from VP_code.utils.data_util import img2tensor, paired_random_crop, augment
    from VP_code.data.Data_Degradation.util import degradation_video_list_4, transfer_1, transfer_2, degradation_video_list_4_one_channel
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
    from VP_code.utils.util import get_root_logger
    from VP_code.utils.data_util import img2tensor, paired_random_crop, augment
    from VP_code.data.Data_Degradation.util import degradation_video_list_4, transfer_1, transfer_2, degradation_video_list_4_one_channel


def getfilelist(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

def getfilelist_with_length(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append((t,len(os.listdir(dir))))

    all_file.sort(key = operator.itemgetter(0))
    return all_file

def getfolderlist(file_path):

    all_folder = []
    for dir,folder,file in os.walk(file_path):
        if len(file)==0:
            continue
        rerank = sorted(file)
        t = "%s/%s"%(dir,rerank[0])
        if t.endswith('.avi'):
            continue
        all_folder.append((t,len(file)))
    
    all_folder.sort(key = operator.itemgetter(0))
    # all_folder = sorted(all_folder)
    return all_folder


def convert_to_L(img):

    frame_pil = transfer_1(img)
    frame_cv2 = transfer_2(frame_pil.convert("RGB"))

    return frame_cv2


def resize_256_short_side(img):
    width, height = img.size

    if width<height:
        new_height =  int (256 * height / width)
        new_width = 256
    else:
        new_width =  int (256 * width / height)
        new_height = 256
    
    return img.resize((new_width,new_height),resample=Image.BILINEAR)


def resize_368_short_side(img):
    
    frame_pil = transfer_1(img)

    width, height = frame_pil.size

    if width<height:
        new_height =  int (368 * height / width)
        new_height = new_height // 16 * 16
        new_width = 368
    else:
        new_width =  int (368 * width / height)
        new_width = new_width // 16 * 16
        new_height = 368
    
    frame_pil = frame_pil.resize((new_width,new_height),resample=Image.BILINEAR)
    return transfer_2(frame_pil.convert("RGB"))


def resize_704_short_side(img):
    
    frame_pil = transfer_1(img)

    width, height = frame_pil.size

    if width<height:
        new_height =  int (704 * height / width)
        new_height = new_height // 16 * 16
        new_width = 704 
    else:
        new_width =  int (704 * width / height)
        new_width = new_width // 16 * 16
        new_height = 704
    
    frame_pil = frame_pil.resize((new_width,new_height),resample=Image.BILINEAR)
    return transfer_2(frame_pil.convert("RGB"))


class Film_dataset_1(data.Dataset): ## 1 for REDS dataset

    def __init__(self, data_config):
        super(Film_dataset_1, self).__init__()
        
        self.data_config = data_config
        
        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)

        self.channels = data_config.get('channels', 3)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

        if self.is_train:

            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)
        
        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)
            # self.lq_frames = []
            # self.gt_frames = []
            # for i in range(len(self.lq_folders))
            #     val_frame_list_this = sorted(os.listdir(self.lq_folders[i]))
            #     first_frame_name = val_frame_list_this[0]
            #     clip_length = len(val_frame_list_this)
            #     self.lq_frames.append((os.path.join(self.lq_folders[i],f'{first_frame_name:08d}.png'),clip_length))
            #     self.gt_frames.append((os.path.join(self.gt_folders[i],f'{first_frame_name:08d}.png'),clip_length))

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            # ensure not exceeding the borders
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # each clip has 100 frames starting from 0 to 99. TODO: if the training clip is not 100 frames [√]
            # Training start frames should be 0
            while (start_frame_idx < 0) or (end_frame_idx > current_len-1):
                center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')
        else:

            frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id, frame in enumerate(frame_list):

            # img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            if self.is_train:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt = cv2.imread(img_gt_path)
                img_gt = img_gt / 255.
                img_gts.append(img_gt)
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
                img_gt = cv2.imread(img_gt_path)
                img_gt = img_gt / 255.
                img_gts.append(img_gt)
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            if self.channels == 3:
                img_lqs, img_gts = degradation_video_list_4(img_gts, texture_url=self.data_config['texture_template'])
            elif self.channels == 1:
                img_lqs, img_gts = degradation_video_list_4_one_channel(img_gts, texture_url=self.data_config['texture_template'])
            else:
                raise NotImplementedError('Image channels should be 1 or 3.');
        else:
            for i in range(len(img_gts)):
                if img_gts[i].shape[0] != 368:
                    img_gts[i] = resize_368_short_side(img_gts[i])
                    img_lqs[i] = resize_368_short_side(img_lqs[i])
                # img_gts[i] = resize_704_short_side(img_gts[i])
                # img_lqs[i] = resize_704_short_side(img_lqs[i])
                if self.channels == 1:
                    img_gts[i] = convert_to_L(img_gts[i])[:, :, :1]
                    img_lqs[i] = convert_to_L(img_lqs[i])[:, :, :1]

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        if self.data_config['normalizing']:
            if self.channels == 3:
                transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            elif self.channels == 1:
                transform_normalize=transforms.Normalize((0.5, ),(0.5, ))
            for i in range(len(img_results)):
                img_results[i]=transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'name_list': new_clip_sequence}

    def __len__(self):
        return len(self.lq_frames)
