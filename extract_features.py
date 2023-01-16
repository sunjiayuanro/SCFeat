import os
import cv2
import datetime
import shutil
import logging
import yaml
import importlib
import numpy as np
import time
from path import Path
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import SCFeat
import dataloader
import dataloader.data_utils as dutils
from SCFeat.preprocess_utils import *
import SCFeat.preprocess_utils as putils

from tqdm import tqdm
import colorlog
from PIL import Image as Im

from torchvision.transforms import Resize

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Extractor(ABC):
    def __init__(self, args):
        self.args       = args
        self.last_batch = None

        with open(self.args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.save_root  = Path('./ckpts/{}'.format(self.config['output_root']))
        self.logfile    = self.save_root/'logging_file.txt'
        self.desc_root  = self.save_root/'desc'
        self.img_root   = self.save_root/'image'
        self.sift_kp    = self.config['use_sift']

        if 'save_npz' in list(self.config.keys()):
            self.save_npz = self.config['save_npz']
        else:
            self.save_npz = True

        ckpt_path   = Path(self.config['load_path'])
        cfg_path    = ckpt_path.dirname()/'config.yaml'
        
        with open(cfg_path, 'r') as f:
            pre_conf = yaml.load(f, Loader=yaml.FullLoader)
        self.config['model_config'].update(pre_conf['model_config'])
        if 'model' in list(pre_conf.keys()):
            self.config['model'] = pre_conf['model']

        self.set_device()
        self.set_folder_and_logger()

        ##  model
        if 'model' in list(self.config.keys()):
            tmp_model = getattr(SCFeat, self.config['model'])
            self.model = tmp_model(self.config['model_config'], self.device)
        else:
            self.logger.error('no such model')
            exit(1)
        if self.multi_gpu:
            self.model.set_parallel(self.args.local_rank)

        self.model.load_checkpoint(self.config['load_path'])
        self.model.set_eval()

        if not self.config['use_sift']:
            self.detector = getattr(putils, self.config['detector'])
            self.logger.info('use {} to detect keypoints'.format(self.config['detector']))
        else:
            self.logger.info('use sift keypoints')

        ##  dataloader
        dataset = getattr(dataloader, self.config['data'])
        extract_dataset = dataset(configs=self.config['data_config_extract'])
        if self.multi_gpu:
            extract_sampler = torch.utils.data.distributed.DistributedSampler(extract_dataset)
        else:
            extract_sampler = None
        self.extract_loader = torch.utils.data.DataLoader(extract_dataset, batch_size=self.config['data_config_extract']['batch_size'], 
                                                       shuffle=False, num_workers=self.config['data_config_extract']['workers'], 
                                                       collate_fn=self.my_collate, sampler=extract_sampler)

    def my_collate(self, batch, _use_shared_memory=True):
        """Puts each data field into a tensor with outer dimension batch size.
        Copied from https://github.com/pytorch in torch/utils/data/_utils/collate.py
        """
        import re
        error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            out = None
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            assert elem_type.__name__ == 'ndarray'
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            batch = [torch.from_numpy(b) for b in batch]
            try:
                return torch.stack(batch, 0)
            except RuntimeError:
                return batch
        elif batch[0] is None:
            return list(batch)
        elif isinstance(batch[0], int):
            return torch.LongTensor(batch)
        elif isinstance(batch[0], float):
            return torch.DoubleTensor(batch)
        elif isinstance(batch[0], str):
            return batch
        elif isinstance(batch[0], dict):
            return {key: self.my_collate([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [self.my_collate(samples) for samples in transposed]

        raise TypeError((error_msg.format(type(batch[0]))))

    def set_device(self):

        if torch.cuda.device_count() == 0:
            self.device = torch.device("cpu")
            self.output_flag=True
            self.multi_gpu = False
            print('use CPU for extraction')
        elif torch.cuda.device_count() == 1:
            self.device = torch.device("cuda")
            self.output_flag=True
            self.multi_gpu = False
            print('use a single GPU for extraction')
        else:
            self.device = torch.device("cuda", self.args.local_rank)
            self.multi_gpu = True
            dist.init_process_group(backend='nccl') 
            # torch.autograd.set_detect_anomaly(True) # for debug
            if self.args.local_rank == 0:
                self.output_flag=True
                print('use {} GPUs for extraction'.format(torch.cuda.device_count()))
            else:
                self.output_flag=False

    def set_folder_and_logger(self):
        if self.output_flag:
            if not os.path.exists(self.save_root) :
                self.save_root.makedirs_p()
            else:
                # if path exsists, quit to make sure that the previous setting.txt would not be overwritten
                if self.config['data'] == 'ETH_LFB' or self.config['data'] == 'IMC_eval':
                    pass 
                else:
                    raise "The save path is already exists, please change the output_root in config" 
            print('=> will save everything to {}'.format(self.save_root))
            # shutil.copy(self.args.config, self.save_root/'config.yaml')
            with open(self.save_root/'config.yaml', 'w') as fout:
                yaml.dump(self.config, fout)
            self.logfile.touch()

            if not os.path.exists(self.desc_root) :
                self.desc_root.makedirs_p()
            if not os.path.exists(self.img_root) :
                self.img_root.makedirs_p()

        while not os.path.exists(self.logfile):
            time.sleep(0.5)
            continue

        self.logger = logging.getLogger()
        if self.output_flag:
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.logfile, mode='a')
            fh.setLevel(logging.DEBUG)

            # ch = logging.StreamHandler()
            ch = TqdmHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter("%(asctime)s - gpu {} - %(levelname)s: %(message)s".format(self.args.local_rank))
            fh.setFormatter(formatter)
            # ch.setFormatter(formatter)
            ch.setFormatter(colorlog.ColoredFormatter(
                "%(asctime)s - gpu {} - %(levelname)s: %(message)s".format(self.args.local_rank),
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'SUCCESS:': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'},))

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        else:
            self.logger.setLevel(logging.ERROR)
            fh = logging.FileHandler(self.logfile, mode='a')
            fh.setLevel(logging.ERROR)

            ch = logging.StreamHandler()
            ch.setLevel(logging.ERROR)

            formatter = logging.Formatter("%(asctime)s - gpu {} - %(levelname)s: %(message)s".format(self.local_rank))
            fh.setFormatter(formatter)
            # ch.setFormatter(formatter)
            ch.setFormatter(colorlog.ColoredFormatter(
                "%(asctime)s - gpu {} - %(levelname)s: %(message)s".format(self.args.local_rank),
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'white',
                    'SUCCESS:': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'},))

            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        # logger.info('test logger')  

    def findthr(self, tensor, thr):
        tensor_np = tensor.cpu().numpy().reshape(-1,1)
        max_val = np.percentile(tensor_np, thr)
        return max_val

    def save_imgs(self, inputs, outputs, processed, idx):
        local_point = outputs['local_point']
        message = "\nlocal_min:{:.3f} max:{:.3f} ".format(
            local_point.min(), local_point.max())

        save_path = self.img_root/inputs['name1'][0]
        name = save_path.name.split('.')[0]
        save_path = save_path.dirname()
        if not save_path.exists():
            save_path.makedirs_p()

        bi, ci, hi, wi = inputs['im1'].shape
        bo, co, ho, wo = local_point.shape
        if hi != ho or wi != wo:
            local_point = F.interpolate(local_point, (hi, wi))
        bi, hi, wi, ci = inputs['im1_ori'].shape

        pad = inputs['pad1']
        if pad[3] != 0:
            local_point = local_point[:,:,:-pad[3],:]
        if pad[1] != 0:
            local_point = local_point[:,:,:,:-pad[1]]
        local_point1 = local_point[:,0,:,:]

        #local_point1 = local_point1/self.findthr(local_point1, 100*self.config['local_thr'])
        local_point1 = local_point1.clamp(0.5,1)

        #local_point1 = dutils.tensor2array(local_point1.squeeze())[:3,:,:].transpose(1,2,0)
        #local_point1 = Im.fromarray((255*local_point1).astype(np.uint8))
        local_point1 = dutils.tensor2array(local_point1.squeeze())[:3,:,:].transpose(1,2,0)
        local_point1 = 255*local_point1
        local_point1 = Im.fromarray(local_point1.astype(np.uint8))
        local_point1.save(save_path/'{:>05d}_score_map.jpg'.format(idx))

        imgs_with_kps = inputs['im1_ori'].squeeze().cpu().numpy().astype(np.uint8)
        # imgs_with_kps = cv2.cvtColor(imgs_with_kps, cv2.COLOR_RGB2BGR)
        color = (0,255,0)
        for kp in processed['kpt']:
            kp = (kp[0], kp[1])
            cv2.circle(imgs_with_kps, kp, radius=2, color=color, thickness=-1)
        imgs_with_kps = cv2.cvtColor(imgs_with_kps, cv2.COLOR_BGR2RGB)
        imgs_with_kps = Im.fromarray(imgs_with_kps)
        imgs_with_kps.save(save_path/'{:>05d}_image_with_kp.jpg'.format(idx))

        return message

    def save_desc(self, inputs, outputs, processed, multiscale=False):
        kpt         = processed['kpt']
        feat_f      = processed['desc']
        kp_score    = processed['kp_score']

        name = inputs['name1'][0]#.replace('ppm','wsf')
        save_path = self.desc_root/name
        
        if not save_path.dirname().exists():
            save_path.dirname().makedirs_p()

        message = "\nkpts: {}".format(kpt.shape[0])

        if self.save_npz:
            desc = None
            if multiscale:
                desc = feat_f
            else:
                desc = feat_f.squeeze(0).detach().cpu().numpy()
            scores = kp_score.squeeze(0).detach().cpu().numpy()
            with open(save_path + '.{}'.format(self.config['postfix']), 'wb') as output_file:
                np.savez(output_file, keypoints=kpt, scores=scores, descriptors=desc)

        return message

    def process(self, inputs, outputs):
        desc_f = outputs['desc_map']
        # desc_c = outputs['desc_map'][0]
        name = inputs['name1'][0]

        b,c,h,w = inputs['im1'].shape

        if self.sift_kp:
            coords = inputs['coord1']
            coord_n = normalize_coords(coords, h, w)
            kp_score = torch.ones_like(coord_n)[:,:,:1]
        else:
            if self.config['data'] == 'Aachen_Day_Night':
                cur_name_split = name.split('/')
                if cur_name_split[0] == 'query':
                    coord_n, kp_score = self.detector(outputs['local_point'], **self.config['detector_config_query'])
                else:
                    coord_n, kp_score = self.detector(outputs['local_point'], **self.config['detector_config'])
            else:
                coord_n, kp_score = self.detector(outputs['local_point'], **self.config['detector_config'])

            coords = denormalize_coords(coord_n, h, w)

        feat_f = sample_feat_by_coord(desc_f, coord_n, self.config['loss_distance']=='cos')
        # feat_c = sample_feat_by_coord(desc_c, coord_n, self.config['loss_distance']=='cos')
        # desc = torch.cat((feat_c, feat_f), -1)
        kpt = coords.cpu().numpy().squeeze(0)

        # scale for inloc
        if 'scale' in list(inputs.keys()):
            kpt = kpt*inputs['scale'].cpu().numpy()

        return {
            'kpt':  kpt,
            'desc': feat_f,
            'kp_score': kp_score
            }

    @torch.no_grad()
    def extract(self):
        bar = tqdm(self.extract_loader, total=int(len(self.extract_loader)), ncols=80)
        
        color = np.array(range(256)).astype(np.float)[None,:].repeat(30, axis=0)
        color = np.concatenate([np.zeros((30,20)),255*np.ones((30,20)),color], axis=1)
        color = dutils.tensor2array(torch.tensor(color))[:3,:,:].transpose(1,2,0)
        color = Im.fromarray((255*color).astype(np.uint8))
        color.save(self.img_root/'0_colorbar.jpg')
        name_list = ''
        
        for idx, inputs in enumerate(bar):
            
            for key, val in inputs.items():
                if key == 'name1' or key == 'pad1':
                    continue
                inputs[key] = val.to(self.device)
            message = inputs['name1'][0]
          
            outputs = self.model.extract(inputs['im1'])
            
            processed = self.process(inputs, outputs)
            if self.config['output_desc']:
                message += self.save_desc(inputs, outputs, processed)
            if self.config['output_img']:
                message += self.save_imgs(inputs, outputs, processed, idx)
            self.logger.info(message)
            name_list += '{} {}\n'.format(idx, inputs['name1'][0])
            torch.cuda.empty_cache()
        with open(self.img_root/'name_list.txt', 'w') as f:
            f.write(name_list)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str, default='./configs/extract.yaml')
args = parser.parse_args()
extractor = Extractor(args)
extractor.extract()