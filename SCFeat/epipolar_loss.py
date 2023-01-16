import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .preprocess_utils import *


class EpipolarLoss_Coarse2Fine(nn.Module):
    def __init__(self, configs, device=None):
        super(EpipolarLoss_Coarse2Fine, self).__init__()
        self.__lossname__ = 'EpipolarLoss_Line2Window'
        
        self.config = configs

        self.w_ec   = self.config['w_epipolar_coarse']
        self.w_ef   = self.config['w_epipolar_fine']
        self.w_cc   = self.config['w_cycle_coarse']
        self.w_cf   = self.config['w_cycle_coarse']
        self.w_std  = self.config['w_std']

    def homogenize(self, coord):
        coord = torch.cat((coord, torch.ones_like(coord[:, :, [0]])), -1)
        return coord

    def set_weight(self, std, mask=None, regularizer=0.0):
        if self.config['std']:
            inverse_std = 1. / torch.clamp(std+regularizer, min=1e-10)
            weight = inverse_std / torch.mean(inverse_std)
            weight = weight.detach()  # Bxn
        else:
            weight = torch.ones_like(std)

        if mask is not None:
            weight *= mask.float()
            weight /= (torch.mean(weight) + 1e-8)
        return weight

    def epipolar_cost(self, coord1, coord2, fmatrix):
        coord1_h = self.homogenize(coord1).transpose(1, 2)
        coord2_h = self.homogenize(coord2).transpose(1, 2)
        # print(coord1_h.shape,fmatrix.shape)
        epipolar_line = fmatrix.bmm(coord1_h)  # Bx3xn
        epipolar_line_ = epipolar_line / torch.clamp(torch.norm(epipolar_line[:, :2, :], dim=1, keepdim=True), min=1e-8)
        essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_, dim=1))  # Bxn
        return essential_cost

    def epipolar_loss(self, coord1, coord2, fmatrix, weight):
        essential_cost = self.epipolar_cost(coord1, coord2, fmatrix)
        loss = torch.mean(weight * essential_cost)
        return loss
    
    def idc_loss(self, desc1, desc2, not_search_mask):

        mask = torch.abs(not_search_mask - 1)
        d1 = self.compute_dist(desc1, desc1)
        d1 = d1 * mask

        d2 = self.compute_dist(desc2, desc2)
        d2 = d2 *  mask

        loss = self.smooth_l1(d1, d2)

        return loss

    def cycle_consistency_loss(self, coord1, coord1_loop, weight, th=40):
        '''
        compute the cycle consistency loss
        :param coord1: [batch_size, n_pts, 2]
        :param coord1_loop: the predicted location  [batch_size, n_pts, 2]
        :param weight: the weight [batch_size, n_pts]
        :param th: the threshold, only consider distances under this threshold
        :return: the cycle consistency loss value
        '''
        distance = torch.norm(coord1 - coord1_loop, dim=-1)
        distance_ = torch.zeros_like(distance)
        distance_[distance < th] = distance[distance < th]
        loss = torch.mean(weight * distance_)
        return loss

    def forward(self, inputs, outputs, processed):
        coord2_ec   = processed['coord2_ec']
        coord2_ef   = processed['coord2_ef']
        coord1_lc   = processed['coord1_lc']
        coord1_lf   = processed['coord1_lf']
        std_c       = processed['std_c']
        std_f       = processed['std_f']
        std_lc      = processed['std_lc']
        std_lf      = processed['std_lf']
        
        im1 = Variable(inputs['im1'])
        im_size = im1.size()[2:]
        shorter_edge, longer_edge = min(im_size), max(im_size)

        coord1 = inputs['coord1'][:,:,:2].cuda()
        fmatrix = inputs['F1'].cuda()
        
        # print(coord1.shape,inputs['F1'], coord2_ec.shape)
        epipolar_cost_c = self.epipolar_cost(coord1, coord2_ec, fmatrix)
        # only add fine level loss if the coarse level prediction is close enough to gt epipolar line
        mask_ctof = (epipolar_cost_c < (shorter_edge * self.config['window_size']))
        # only add cycle consistency loss if the coarse level prediction is close enough to gt epipolar line
        mask_epip_c = (epipolar_cost_c < (shorter_edge * self.config['th_epipolar']))
        mask_cycle_c = (epipolar_cost_c < (shorter_edge * self.config['th_cycle']))

        epipolar_cost_f = self.epipolar_cost(coord1, coord2_ef, fmatrix)
        # only add cycle consistency loss if the fine level prediction is close enough to gt epipolar line
        mask_epip_f = (epipolar_cost_f < (shorter_edge * self.config['th_epipolar']))
        mask_cycle_f = (epipolar_cost_f < shorter_edge * self.config['th_cycle'])

        weight_c = self.set_weight(std_c, mask=mask_epip_c)
        weight_f = self.set_weight(std_f, mask=mask_epip_f*mask_ctof)

        eloss_c = torch.mean(epipolar_cost_c * weight_c) / longer_edge
        eloss_f = torch.mean(epipolar_cost_f * weight_f) / longer_edge

        weight_cycle_c = self.set_weight(std_c * std_lc, mask=mask_cycle_c)
        weight_cycle_f = self.set_weight(std_f * std_lf, mask=mask_cycle_f)

        closs_c = self.cycle_consistency_loss(coord1, coord1_lc, weight_cycle_c) / longer_edge
        closs_f = self.cycle_consistency_loss(coord1, coord1_lf, weight_cycle_f) / longer_edge

        valid_mask = mask_epip_f*mask_ctof
        invalid_mask = (~valid_mask).float()
        not_search_mask = torch.bmm(invalid_mask.unsqueeze(-1), invalid_mask.unsqueeze(1))
    
        # idcloss_f = self.idc_loss(processed['feat1_fine'], processed['feat2_fine'], not_search_mask)

        loss = self.w_ec * eloss_c + self.w_ef * eloss_f + self.w_cc * closs_c + self.w_cf * closs_f

        std_loss = torch.mean(std_c) + torch.mean(std_f)
        loss += self.w_std * std_loss

        # loss += 0.01 * idcloss_f * processed['epoch']

        return loss, {}

class EpipolarLoss_Line2Window(nn.Module):
    def __init__(self, configs, device=None):
        super(EpipolarLoss_Line2Window, self).__init__()
        self.__lossname__ = 'EpipolarLoss_Line2Window'
        
        self.config = configs

        self.w_g = self.config['weight_grid']
        self.w_w = self.config['weight_window']

    def epipolar_cost(self, coord1, coord2, fmatrix, im_size):
        coord1_h = homogenize(coord1).transpose(1, 2)
        coord2_h = homogenize(coord2).transpose(1, 2)
        epipolar_line = fmatrix.bmm(coord1_h)  # Bx3xn
        epipolar_line_ = epipolar_line / torch.clamp(torch.norm(epipolar_line[:, :2, :], dim=1, keepdim=True), min=1e-8)
        essential_cost = torch.abs(torch.sum(coord2_h * epipolar_line_, dim=1))  # Bxn
        return essential_cost


    def set_weight(self, inverse_std, mask=None, regularizer=0.0):
        if self.config['use_std_as_weight']:
            # inverse_std = 1. / torch.clamp(std+regularizer, min=1e-10)
            weight = inverse_std / torch.mean(inverse_std)
            weight = weight.detach()  # Bxn
        else:
            weight = torch.ones_like(std)

        if mask is not None:
            weight *= mask.float()
            weight /= (torch.mean(weight) + 1e-8)
        return weight 

    def forward(self, inputs, outputs, processed):
        coord1      = processed['coord1']
        coord2      = processed['coord2']
        temperature = processed['temperature']

        feat1g_corloc = processed['feat1g_corloc']
        feat2g_corloc = processed['feat2g_corloc']
        feat1w_corloc = processed['feat1w_corloc']
        feat2w_corloc = processed['feat2w_corloc']

        feat1g_std = processed['feat1g_std']
        feat2g_std = processed['feat2g_std']
        feat1w_std = processed['feat1w_std']
        feat2w_std = processed['feat2w_std']

        Fmat1 = inputs['F1']
        Fmat2 = inputs['F2']
        im_size1 = inputs['im1'].size()[2:]
        im_size2 = inputs['im2'].size()[2:]
        shorter_edge, longer_edge = min(im_size1), max(im_size1)

        cost_g1 = self.epipolar_cost(coord1, feat1g_corloc, Fmat1, im_size1)
        cost_w1 = self.epipolar_cost(coord1, feat1w_corloc, Fmat1, im_size1)

        cost_g2 = self.epipolar_cost(coord2, feat2g_corloc, Fmat2, im_size2)
        cost_w2 = self.epipolar_cost(coord2, feat2w_corloc, Fmat2, im_size2)

        # filter out the large values, similar to CAPS
        mask_g1 = cost_g1 < (shorter_edge*self.config['grid_cost_thr'])
        mask_w1 = cost_w1 < (shorter_edge*self.config['win_cost_thr'])
        mask_g2 = cost_g2 < (shorter_edge*self.config['grid_cost_thr'])
        mask_w2 = cost_w2 < (shorter_edge*self.config['win_cost_thr'])

        if 'valid_epi1' in list(processed.keys()):
            mask_g1 = mask_g1 & processed['valid_epi1']
            mask_w1 = mask_w1 & processed['valid_epi1']
            mask_g2 = mask_g2 & processed['valid_epi2']
            mask_w2 = mask_w2 & processed['valid_epi2']
        weight_w1 = 1
        weight_w2 = 1 

        weight_g1 = self.set_weight(1/feat1g_std.clamp(min=1e-10), mask_g1)
        weight_w1 = self.set_weight(weight_w1/feat1w_std.clamp(min=1e-10), mask_w1)
        weight_g2 = self.set_weight(1/feat2g_std.clamp(min=1e-10), mask_g2)
        weight_w2 = self.set_weight(weight_w2/feat2w_std.clamp(min=1e-10), mask_w2)

        loss_g1 = (weight_g1*cost_g1).mean()
        loss_w1 = (weight_w1*cost_w1).mean()
        loss_g2 = (weight_g2*cost_g2).mean()
        loss_w2 = (weight_w2*cost_w2).mean()

        loss = self.w_g*(loss_g1+loss_g2)+self.w_w*(loss_w1+loss_w2)

        percent_g = (mask_g1.sum()/(mask_g1.shape[0]*mask_g1.shape[1]) + mask_g2.sum()/(mask_g2.shape[0]*mask_g2.shape[1]))/2
        percent_w =  (mask_w1.sum()/(mask_w1.shape[0]*mask_w1.shape[1]) + mask_w2.sum()/(mask_w2.shape[0]*mask_w2.shape[1]))/2

        components = {
            'loss_g1':      loss_g1, 
            'loss_w1':      loss_w1, 
            'loss_g2':      loss_g2, 
            'loss_w2':      loss_w2, 
            'percent_g':    percent_g, 
            'percent_w':    percent_w
            }

        return loss, components