import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from path import Path
import os

import SCFeat

class SCFeatModel(ABC):
    """
    SCFeat Model
    """
    def __init__(self, configs, device, no_cuda=None):
        self.config     = configs
        self.device     = device
        self.no_cuda    = no_cuda

        self.align_local_grad       = self.config['align_local_grad']
        self.local_input_elements   = self.config['local_input_elements']
        self.local_with_img         = self.config['local_with_img']
        
        self.parameters = []

        # backbone
        backbone = getattr(SCFeat, self.config['backbone'])
        self.backbone = backbone(**self.config['backbone_config']).to(self.device)
        self.parameters += list(self.backbone.parameters())
        message = "backbone: {}\n".format(self.config['backbone'])

        # localheader
        if 'localheader' in list(self.config.keys()) and self.config['localheader'] != 'None':
            localheader = getattr(SCFeat, self.config['localheader'])
            self.localheader = localheader(self.backbone, **self.config['localheader_config']).to(self.device)
            message += "localheader: {}\n".format(self.config['localheader'])
        else:
            in_channel = self.backbone.out_channels[0]
            self.localheader = SCFeat.DetNet(in_channels=in_channel, out_channels=2).to(self.device)
            message += "localheader: DetNet(else)\n"
        self.parameters += list(self.localheader.parameters())

        self.modules = ['localheader', 'backbone']
        print(message)

    def name(self):
        return 'SCFeatModel'

    def load_checkpoint(self, load_path):
        load_root = Path(load_path)
        model_list = ['backbone', 'localheader']
        for name in model_list:
            model_path = load_root/'{}.pth'.format(name)
            if os.path.exists(model_path):
                print('load {} from checkpoint'.format(name))
            else:
                print('{} does not exist, skipping load'.format(name))
                continue
            model = getattr(self, name)
            model_param = torch.load(model_path)
            model.load_state_dict(model_param)

    def save_checkpoint(self, save_path):
        save_root = Path(save_path)
        model_list = ['backbone', 'localheader']
        for name in model_list:
            model_path = save_root/'{}.pth'.format(name)
            model = getattr(self, name)
            model_param = model.state_dict()
            torch.save(model_param, model_path)

    def set_train(self):
        self.backbone.train()
        self.localheader.train()

    def set_eval(self):
        self.backbone.eval()
        self.localheader.eval()

    def extract(self, tensor):
        feat_maps = self.backbone(tensor)
        local_list = []
        for name in self.local_input_elements:
            local_list.append(feat_maps[name])
        local_input = torch.cat(local_list, dim=1)
        if not self.align_local_grad:
            local_input = local_input.detach()
        if self.local_with_img:
            local_input = [local_input, tensor]
        
        l_map = self.localheader(local_input)
        if l_map.shape[1] == 1:
            local_thr = torch.zeros_like(l_map)
        elif l_map.shape[1] == 2:
            local_thr = l_map[:,1:,:,:]
            l_map = l_map[:,:1,:,:]

        outputs = {
            'local_map':    feat_maps['local_map'],
            'global_map':   feat_maps['global_map'],
            'desc_map':     feat_maps['desc_map'],
            'local_point':  l_map,
            'local_thr':    local_thr,
        }

        return outputs

    def forward(self, inputs):
        for key, val in inputs.items():
            if key in self.no_cuda:
                continue
            inputs[key] = val.to(self.device)

        preds1 = self.extract(inputs['im1'])
        preds2 = self.extract(inputs['im2'])

        return {
            'preds1':   preds1, 
            'preds2':   preds2
            }

