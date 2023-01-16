import os
import datetime
import shutil
import logging
import yaml
import importlib
import time
from path import Path
from abc import ABC, abstractmethod
from PIL import Image as Im
import numpy as np
import torch.nn.functional as F

import torch
from tensorboardX import SummaryWriter

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import SCFeat
import dataloader

from tqdm import tqdm
import cv2
import copy
import matplotlib
import matplotlib.pyplot as plt

class TqdmHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


class Trainer(ABC):
    '''
    the preprocess class for coarse to fine pipeline
    '''
    def __init__(self, args):
        # read the config file
        self.last_batch = None
        self.args = args
        with open(self.args.config, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.save_root = Path('./ckpts/{}'.format(self.config['checkpoint_name']))
        self.logfile = self.save_root/'logging_file.txt'

        # update the model config if there is a checkpoint
        ckpt_path = None
        if 'load_path' in list(self.config.keys()):
            if self.config['load_path'] is not None:
                ckpt_path = Path(self.config['load_path'])
                cfg_path = ckpt_path.dirname()/'config.yaml'
                with open(cfg_path, 'r') as f:
                    pre_conf = yaml.load(f, Loader=yaml.FullLoader)
                self.config['model_config'].update(pre_conf['model_config'])

                if 'model' in list(pre_conf.keys()):
                    self.config['model'] = pre_conf['model']

        self.set_device()
        self.set_folder_and_logger()

        # model
        if 'model' in list(self.config.keys()):
            tmp_model = getattr(SCFeat, self.config['model'])
            self.model = tmp_model(self.config['model_config'], self.device, self.config['no_cuda'])
        else:
            self.model = SCFeat.SCFeatModel(self.config['model_config'], self.device, self.config['no_cuda'])
        parameters = []
        for module_name, module_lr in zip(self.config['optimal_modules'], self.config['optimal_lrs']):
            tmp_module = getattr(self.model, module_name)
            parameters.append({'params':tmp_module.parameters(), 'lr':module_lr})
        self.all_optimized_modules = self.config['optimal_modules']
        for module_name in self.model.modules:
            if module_name not in self.all_optimized_modules:
                tmp_module = getattr(self.model, module_name)
                for p in tmp_module.parameters():
                    p.requires_grad = False
        if ckpt_path is not None:
            self.logger.info('load checkpoint from {}'.format(ckpt_path))
            self.model.load_checkpoint(ckpt_path)
        if self.multi_gpu:
            self.model.set_parallel(self.args.local_rank)

        # losses
        if 'preprocess_train' in list(self.config.keys()):
            tmp_model = getattr(SCFeat, self.config['preprocess_train'])
            self.preprocess = tmp_model(self.config['preprocess_train_config'], self.device).to(self.device)
            self.skip_preprocess = False
        else:
            self.preprocess = SCFeat.Preprocess_Skip().to(self.device)
            self.skip_preprocess = True

        self.losses = []
        self.losses_weight = []
        for loss_name, loss_weight in zip(self.config['losses'], self.config['losses_weight']):
            loss_module = getattr(SCFeat, loss_name)
            self.losses.append(loss_module(self.config['{}_config'.format(loss_name)], self.device).to(self.device))
            self.losses_weight.append(float(loss_weight))
            if hasattr(self.losses[-1], 'load_checkpoint'):
                if ckpt_path is not None: 
                    self.losses[-1].load_checkpoint(ckpt_path)
                parameters.append({'params':self.losses[-1].parameters()})

        # optimizer
        self.logger.info(parameters)
        self.logger.info(self.all_optimized_modules)
        tmp_optimizer = getattr(torch.optim, self.config['optimizer'])
        self.optimizer = tmp_optimizer(parameters)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=self.config['lr_decay_step'],
                                                         gamma=self.config['lr_decay_factor'])
        self.logger.info(self.config['optimizer'])

        #  dataloader
        dataset = getattr(dataloader, self.config['data'])
        train_dataset = dataset(configs=self.config['data_config_train'], is_train=True)
        if self.multi_gpu:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['data_config_train']['batch_size'], 
                                                       shuffle= ~self.multi_gpu, num_workers=self.config['data_config_train']['workers'], 
                                                       collate_fn=self.my_collate, sampler=train_sampler)

        val_dataset = dataset(configs=self.config['val_config']['data_config_val'], is_train=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['val_config']['data_config_val']['batch_size'], 
                                               shuffle= self.config['val_config']['data_config_val']['shuffle'], 
                                               num_workers=self.config['val_config']['data_config_val']['workers'], 
                                               collate_fn=self.my_collate)
        val_iter = iter(self.cycle(val_loader))


        self.val_data = next(val_iter)
        while True:
            if self.val_data == None:
                self.val_data = next(val_iter)
                continue
            break
        # print('next')
        del val_dataset, val_loader, val_iter
        with open(self.save_root/'val_data.npz', 'wb') as out_f:
            np.savez(out_f, val_data=self.val_data)

    def my_collate(self, batch):
        ''' Puts each data field into a tensor with outer dimension batch size '''
        batch = list(filter(lambda b: b is not None, batch))
        # check batch
        if batch == None:
            return None
        if len(batch) == 0:
            if self.last_batch == None:
                return None
            return torch.utils.data.dataloader.default_collate(self.last_batch)
        self.last_batch = batch
        
        return torch.utils.data.dataloader.default_collate(batch)

    def cycle(self, iterable):
      while True:
          for x in iterable:
              yield x

    def set_device(self):
        if torch.cuda.device_count() == 0:
            self.device = torch.device("cpu")
            self.output_flag=True
            self.multi_gpu = False
            print('use CPU for training')
        elif torch.cuda.device_count() == 1:
            self.device = torch.device("cuda")
            self.output_flag=True
            self.multi_gpu = False
            self.args.local_rank = 0
            print('use a single GPU for training')
        else:
            self.device = torch.device("cuda", self.args.local_rank)
            self.multi_gpu = True
            dist.init_process_group(backend='nccl') 
            # torch.autograd.set_detect_anomaly(True) # for debug
            if self.args.local_rank == 0:
                self.output_flag=True
                print('use {} GPUs for training'.format(torch.cuda.device_count()))
            else:
                self.output_flag=False

    def set_folder_and_logger(self):
        if self.output_flag:
            if not os.path.exists(self.save_root) :
                self.save_root.makedirs_p()
            else:
                # TODO
                raise "The save path is already exists, please update the folder name" 
            print('=> will save everything to {}'.format(self.save_root))
            with open(self.save_root/'config.yaml', 'w') as fout:
                yaml.dump(self.config, fout)
            self.logfile.touch()

            self.writer = SummaryWriter(self.save_root)

        self.logger = logging.getLogger()

        # color settings
        BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
        RESET_SEQ = "\033[0m"
        COLOR_SEQ = "\033[1;%dm"
        BOLD_SEQ = "\033[1m"

        def formatter_message(message, use_color = True):
            if use_color:
                message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
            else:
                message = message.replace("$RESET", "").replace("$BOLD", "")
            return message

        COLORS = {
            'WARNING': YELLOW,
            'INFO': CYAN,
            'DEBUG': BLUE,
            'CRITICAL': YELLOW,
            'ERROR': RED
        }

        class ColoredFormatter(logging.Formatter):
            def __init__(self, msg, use_color = True):
                logging.Formatter.__init__(self, msg)
                self.use_color = use_color
         
            def format(self, record):
                levelname = record.levelname
                if self.use_color and levelname in COLORS:
                    levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
                    record.levelname = levelname_color
                return logging.Formatter.format(self, record)

        msg = "%(asctime)s-gpu {}-%(levelname)s: %(message)s".format(self.args.local_rank)
        formatter = logging.Formatter(msg)
        color_formatter = ColoredFormatter(formatter_message(msg, True))

        if self.output_flag:
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler(self.logfile, mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)

            ch = TqdmHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(color_formatter)
        else:
            self.logger.setLevel(logging.ERROR)
            fh = logging.FileHandler(self.logfile, mode='a')
            fh.setLevel(logging.ERROR)
            fh.setFormatter(formatter)

            ch = TqdmHandler()
            ch.setLevel(logging.ERROR)
            ch.setFormatter(color_formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def save_errors(self, inputs, outputs, losses, loss_items):
        if not os.path.exists(self.save_root/"error.pt"):
            save_dict = {"inputs":inputs, "outputs": outputs,
                    "losses":losses, "loss_items":loss_items}
            torch.save(save_dict, self.save_root/"error.pt")

    def save_loss(self, save_path):
        save_path = Path(save_path)
        for idx in range(len(self.config['losses'])):
            if hasattr(self.losses[idx], 'save_checkpoint'):
                self.losses[idx].save_checkpoint(save_path)

    def train(self):
        batch_size_val = self.val_data['im1'].shape[0]
        epoch_path = self.save_root/'{:>03d}'.format(0)
        epoch_path.makedirs_p()
        self.model.save_checkpoint(epoch_path)
        self.save_loss(epoch_path)

        for epoch in range(self.config['epoch']):
            epoch += 1
            epoch_path = self.save_root/'{:>03d}'.format(epoch)
            epoch_path.makedirs_p()
            batch_path_list = []
            for i in range(batch_size_val):
                batch_path = epoch_path/'{}'.format(i)
                batch_path.makedirs_p()
                batch_path_list.append(batch_path)
            if self.config['epoch_step'] > 0:
                total_steps = self.config['epoch_step']
            else:
                total_steps = len(self.train_loader)
            bar = tqdm(self.train_loader, total=int(total_steps), ncols=80)
            bar.set_description('{}/{} {}/{}'.format(self.config['checkpoint_name'], self.save_root.name, epoch, self.config['epoch']))
            self.model.set_train()
            for idx, inputs in enumerate(bar):
                if inputs == None:
                    continue
                # train
                self.model.set_eval()
                for module in self.config['optimal_modules']:
                    tmp_module = getattr(self.model, module)
                    tmp_module.train()
                outputs = self.model.forward(inputs)
                outputs['epoch'] = epoch
                outputs['iterations'] = int((epoch-1)*total_steps+idx)
                processed = self.preprocess(inputs, outputs)
                if self.skip_preprocess:
                    message = "epoch {} batch {}".format(epoch, idx)
                else:
                    message = "epoch {} batch {} temperature {}".format(epoch, idx, processed['temperature'])
                total_loss = 0
                loss_items = []
                temp_log = {}
                for loss_name, loss_module, loss_weight in zip(self.config['losses'], self.losses, self.losses_weight):
                    tmp_loss, tmp_items = loss_module(inputs, outputs, processed)
                    total_loss += loss_weight*tmp_loss.mean()
                    temp_log[loss_name] = tmp_loss.detach().mean().item()
                    message += "\n  {}:{:.5f}[{:.2f}] (total: {:.5f} ".format(loss_name, loss_weight*tmp_loss.detach().mean().item(), loss_weight, 
                        tmp_loss.detach().mean().item())
                    message += ")"
                    loss_items.append(tmp_items)
                message += '\n'

                # if the loss is nan, skip this batch
                if total_loss != total_loss:
                    self.logger.info(message)
                    self.logger.error("loss is nan in {}, check the error.pt".format(idx))
                    self.save_errors(inputs, outputs, total_loss, loss_items)
                    total_loss.backward()
                    self.optimizer.zero_grad() 
                    continue

                self.optimizer.zero_grad() 
                total_loss.backward()

                if 'localheader' in self.all_optimized_modules:
                    grad_message = 'grad localheader conv1 mean {:.6f} max{:.6f}'.format(self.model.localheader.conv1.weight.grad.mean().item(), 
                        self.model.localheader.conv1.weight.grad.max().item())
                    self.logger.info(grad_message)
                if 'backbone' in self.all_optimized_modules:
                    grad_message = 'grad backbone fusion_conv mean {:.6f} max{:.6f}'.format(self.model.backbone.fusion_conv.weight.grad.mean().item(), 
                        self.model.backbone.fusion_conv.weight.grad.max().item())
                    self.logger.info(grad_message)
                    grad_message = 'grad backbone firstconv mean {:.6f} max{:.6f}'.format(self.model.backbone.firstconv.weight.grad.mean().item(), 
                        self.model.backbone.firstconv.weight.grad.max().item())
                    self.logger.info(grad_message)
                if self.config['grad_clip']:
                    for module_name in self.all_optimized_modules:
                        tmp_module = getattr(self.model, module_name)
                        torch.nn.utils.clip_grad_norm_(tmp_module.parameters(), self.config['clip_norm'])
                    if 'localheader' in self.all_optimized_modules:
                        grad_message = 'grad clipped localheader conv1 mean {:.6f} max{:.6f}'.format(self.model.localheader.conv1.weight.grad.mean().item(), 
                            self.model.localheader.conv1.weight.grad.max().item())
                        self.logger.info(grad_message)
                    if 'backbone' in self.all_optimized_modules:
                        grad_message = 'grad clipped backbone firstconv mean {:.6f} max{:.6f}'.format(self.model.backbone.firstconv.weight.grad.mean().item(), 
                            self.model.backbone.firstconv.weight.grad.max().item())
                        self.logger.info(grad_message)
                self.optimizer.step()
                
                self.logger.info(message)
                if self.output_flag and idx%self.config['log_freq'] == 0:
                    self.writer.add_scalar('losses', total_loss.item(), int((epoch-1)*total_steps+idx))
                    for loss_name in self.config['losses']:
                        self.writer.add_scalar(loss_name, temp_log[loss_name], int((epoch-1)*total_steps+idx))
                    for components in loss_items:
                        for component_name in list(components.keys()):
                            if component_name in self.config['tb_component']:
                                self.writer.add_scalar(component_name, components[component_name], int((epoch-1)*total_steps+idx))
                if self.output_flag and idx%100 == 0:
                    self.model.save_checkpoint(epoch_path)

                torch.cuda.empty_cache()

                if idx>=self.config['epoch_step']:
                    break

            
            self.model.save_checkpoint(epoch_path)
            self.save_loss(epoch_path)
            self.scheduler.step()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--config', type=str, default='./configs/train.yaml')
args = parser.parse_args()
trainer = Trainer(args)
trainer.train()