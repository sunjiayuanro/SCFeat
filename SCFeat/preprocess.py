import torch
import torch.nn as nn
import torch.nn.functional as F
from . import preprocess_utils as putils
from .preprocess_utils import *

class Preprocess_Coarse2Fine(nn.Module):
    '''
    the preprocess class for coarse to fine pipeline
    '''
    def __init__(self, configs, device=None, vis=False):
        super(Preprocess_Coarse2Fine, self).__init__()
        self.__lossname__ = 'Preprocess_Coarse2Fine'
        self.config = configs
        
        self.kps_generator  = getattr(putils, self.config['kps_generator'])
        self.t_base         = self.config['temperature_base']
        self.t_max          = self.config['temperature_max']
        
        if device is not None:
            self.device = device

    def name(self):
        return self.__lossname__

    @staticmethod
    def normalize(coord, h, w):
        '''
        turn the coordinates from pixel indices to the range of [-1, 1]
        :param coord: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: the normalized coordinates [..., 2]
        '''
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord.device).float()
        coord_norm = (coord - c) / c
        return coord_norm

    @staticmethod
    def denormalize(coord_norm, h, w):
        '''
        turn the coordinates from normalized value ([-1, 1]) to actual pixel indices
        :param coord_norm: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: actual pixel coordinates
        '''
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord_norm.device)
        coord = coord_norm * c + c
        return coord

    def ind2coord(self, ind, width):
        ind = ind.unsqueeze(-1)
        x = ind % width
        y = ind // width
        coord = torch.cat((x, y), -1).float()
        return coord

    def get_1nn_coord(self, feat1, featmap2):
        '''
        find the coordinates of nearest neighbor match
        :param feat1: query features, [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the other image
        :return: normalized correspondence locations [batch_size, n_pts, 2]
        '''
        batch_size, d, h, w = featmap2.shape
        feat2_flatten = featmap2.reshape(batch_size, d, h*w).transpose(1, 2)  # Bx(hw)xd
        
        sim = feat1.bmm(feat2_flatten.transpose(1, 2))
        ind2_1nn = torch.max(sim, dim=-1)[1]

        coord2 = self.ind2coord(ind2_1nn, w)
        coord2_n = self.normalize(coord2, h, w)
        return coord2_n
    def get_expected_correspondence_within_window(self, feat1, featmap2, coord2_n, window_size, with_std=False):
        '''
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param coord2_n: normalized center locations [batch_size, n_pts, 2]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations, [batch_size, n_pts, 2], optionally with std
        '''
        batch_size, n_dim, h2, w2 = featmap2.shape
        n_pts = coord2_n.shape[1]
        grid_n = gen_grid(h_min=-window_size, h_max=window_size,
                               w_min=-window_size, w_max=window_size,
                               len_h=int(window_size*h2), len_w=int(window_size*w2))

        grid_n_ = grid_n.repeat(batch_size, 1, 1, 1).cuda()  # Bx1xhwx2
        coord2_n_grid = coord2_n.unsqueeze(-2) + grid_n_  # Bxnxhwx2
        feat2_win = F.grid_sample(featmap2, coord2_n_grid, padding_mode='zeros').permute(0, 2, 3, 1)  # Bxnxhwxd

        feat1 = feat1.unsqueeze(-2)

        prob = compute_prob(feat1.reshape(batch_size*n_pts, -1, n_dim),
                                 feat2_win.reshape(batch_size*n_pts, -1, n_dim)).reshape(batch_size, n_pts, -1)

        expected_coord2_n = torch.sum(coord2_n_grid * prob.unsqueeze(-1), dim=2)  # Bxnx2

        if with_std:
            var = torch.sum(coord2_n_grid**2 * prob.unsqueeze(-1), dim=2) - expected_coord2_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord2_n, std
        else:
            return expected_coord2_n

    def forward(self, inputs, outputs):
        preds1 = outputs['preds1']
        preds2 = outputs['preds2']

        xc1, xf1 = preds1['global_map'], preds1['local_map']
        xc2, xf2 = preds2['global_map'], preds2['local_map']
        h1i, w1i = inputs['im1'].size()[2:]
        h2i, w2i = inputs['im2'].size()[2:]

        temperature = min(self.t_base + outputs['epoch'], self.t_max)

        b, _, hf, wf = xf1.shape
        coord1 = inputs['coord1'][:,:,:2].cuda()
        coord1_n = normalize_coords(coord1, h1i, w1i)
        feat1_coarse = sample_feat_by_coord(xc1, coord1_n)  # Bxnxd
        coord2_ec_n, std_c = get_expected_correspondence_locs(feat1_coarse, xc2, with_std=True)

        # the center locations  of the local window for fine level computation
        coord2_ec_n_ = self.get_1nn_coord(feat1_coarse, xc2) if self.config['use_nn'] else coord2_ec_n
        feat1_fusion = sample_feat_by_coord(xf1, coord1_n)  # Bxnxd
        coord2_ef_n, std_f = self.get_expected_correspondence_within_window(feat1_fusion, xf2, coord2_ec_n_, self.config['window_size'], with_std=True)
                                                                    
        feat2_coarse = sample_feat_by_coord(xc2, coord2_ec_n_)
        coord1_lc_n, std_lc = get_expected_correspondence_locs(feat2_coarse, xc1, with_std=True)

        feat2_fusion = sample_feat_by_coord(xf2, coord2_ef_n)  # Bxnxd
        coord1_lf_n, std_lf = self.get_expected_correspondence_within_window(feat2_fusion, xf1, coord1_n, self.config['window_size'], with_std=True)
        coord2_ec = denormalize_coords(coord2_ec_n, h2i, w2i)
        coord2_ef = denormalize_coords(coord2_ef_n, h2i, w2i)
        coord1_lc = denormalize_coords(coord1_lc_n, h1i, w1i)
        coord1_lf = denormalize_coords(coord1_lf_n, h1i, w1i)

        return {
            'coord2_ec':    coord2_ec,
            'coord2_ef':    coord2_ef,
            'coord1_lc':    coord1_lc, 
            'coord1_lf':    coord1_lf,
            'std_c':        std_c, 
            'std_f':        std_f,
            'std_lc':       std_lc, 
            'std_lf':       std_lf,
            'temperature':  temperature,
            'epoch':        outputs['epoch'],
            'feat1_fusion':   feat1_fusion, 
            'feat2_fusion':   feat2_fusion,
            }

class Preprocess_Line2Window(nn.Module):
    '''
    the preprocess class for grid-with-line pipeline
    '''
    def __init__(self, configs, device=None, vis=False):
        super(Preprocess_Line2Window, self).__init__()
        self.__lossname__ = 'Preprocess_Line2Window'

        self.config = configs

        self.kps_generator  = getattr(putils, self.config['kps_generator'])
        self.t_base         = self.config['temperature_base']
        self.t_max          = self.config['temperature_max']

        if device is not None:
            self.device = device

    def name(self):
        return self.__lossname__

    def forward(self, inputs, outputs):
        preds1 = outputs['preds1']
        preds2 = outputs['preds2']

        xf1 = preds1['local_map']
        xf2 = preds2['local_map']
        h1i, w1i = inputs['im1'].size()[2:]
        h2i, w2i = inputs['im2'].size()[2:]
        
        b, _, hf, wf = xf1.shape

        temperature = min(self.t_base + outputs['epoch'], self.t_max)

        coord1_n, coord2_n, score1, score2 = self.kps_generator(inputs, outputs, **self.config['kps_generator_config'])

        coord1 = denormalize_coords(coord1_n.reshape(b,-1,2), h1i, w1i)
        coord2 = denormalize_coords(coord2_n.reshape(b,-1,2), h2i, w2i)

        feat1_fusion = sample_feat_by_coord(xf1, coord1_n.reshape(b,-1,2), self.config['loss_distance']=='cos')
        feat2_fusion = sample_feat_by_coord(xf2, coord2_n.reshape(b,-1,2), self.config['loss_distance']=='cos')

        cos_sim = feat1_fusion @ feat2_fusion.transpose(1,2) # bxmxn
        feat1g_corloc = (F.softmax(temperature*cos_sim, dim=2)).unsqueeze(-1)*coord2.reshape(b,-1,2).unsqueeze(1) #bxmxnx2
        feat1g_corloc = feat1g_corloc.sum(2) #bxmx2
        feat2g_corloc = (F.softmax(temperature*cos_sim, dim=1)).unsqueeze(-1)*coord1.reshape(b,-1,2).unsqueeze(2) #bxmxnx2
        feat2g_corloc = feat2g_corloc.sum(1) #bxnx2


        with torch.no_grad():
            if self.config['use_nn_grid']:
                _, max_idx1 = cor_mat.max(2)
                feat1g_corloc_n = coord2_n.reshape(b,-1,2).gather(dim=1, index=max_idx1[:,:,None].repeat(1,1,2))
                _, max_idx2 = cor_mat.max(1)
                feat2g_corloc_n = coord1_n.reshape(b,-1,2).gather(dim=1, index=max_idx2[:,:,None].repeat(1,1,2))
            else:
                feat1g_corloc_n = normalize_coords(feat1g_corloc, h2i, w2i)
                feat2g_corloc_n = normalize_coords(feat2g_corloc, h1i, w1i)

        feat1g_std = (F.softmax(temperature*cos_sim, dim=2)).unsqueeze(-1)*(coord2_n.reshape(b,1,-1,2)**2)
        feat1g_std = feat1g_std.sum(2) - (feat1g_corloc_n**2)
        feat1g_std = feat1g_std.clamp(min=1e-6).sqrt().sum(-1) #bxn
        feat2g_std = (F.softmax(temperature*cos_sim, dim=1)).unsqueeze(-1)*(coord1_n.reshape(b,-1,1,2)**2)
        feat2g_std = feat2g_std.sum(1) - (feat2g_corloc_n**2)
        feat2g_std = feat2g_std.clamp(min=1e-6).sqrt().sum(-1) #bxn

        if self.config['use_line_search']:
            feat1_c_corloc_n_, feat1_c_corloc_n_org, valid1, epi_std1 = epipolar_line_search(coord1, inputs['F1'], feat1_fusion, 
                temperature*F.normalize(xf2,p=2.0,dim=1), h2i, w2i, window_size=self.config['window_size'], **self.config['line_search_config'])
            feat2_c_corloc_n_, feat2_c_corloc_n_org, valid2, epi_std2 = epipolar_line_search(coord2, inputs['F2'], feat2_fusion, 
                temperature*F.normalize(xf1,p=2.0,dim=1), h1i, w1i, window_size=self.config['window_size'], **self.config['line_search_config'])
            feat1c_corloc_org = denormalize_coords(feat1_c_corloc_n_org, h2i, w2i)
            feat2c_corloc_org = denormalize_coords(feat2_c_corloc_n_org, h1i, w1i)
        else:
            feat1_c_corloc_n_ = feat1g_corloc_n.detach()
            feat2_c_corloc_n_ = feat2g_corloc_n.detach()
            feat1c_corloc_org = feat1_c_corloc_n_
            feat2c_corloc_org = feat2_c_corloc_n_
            valid1 = torch.ones_like(feat1g_std).bool()
            valid2 = torch.ones_like(feat2g_std).bool()

        feat1w_corloc_n, window_coords_n_1in2, feat1w_std, _ = get_expected_correspondence_within_window(
            feat1_fusion, temperature*F.normalize(xf2,p=2.0,dim=1), feat1_c_corloc_n_, self.config['window_size'], with_std=True)
        feat2w_corloc_n, window_coords_n_2in1, feat2w_std, _ = get_expected_correspondence_within_window(
            feat2_fusion, temperature*F.normalize(xf1,p=2.0,dim=1), feat2_c_corloc_n_, self.config['window_size'], with_std=True)

        feat1w_corloc = denormalize_coords(feat1w_corloc_n, h2i, w2i)
        feat2w_corloc = denormalize_coords(feat2w_corloc_n, h1i, w1i)

        return {
                'coord1':           coord1, 
                'coord2':           coord2,
                'feat1g_corloc':    feat1g_corloc,
                'feat2g_corloc':    feat2g_corloc,
                'feat1w_corloc':    feat1w_corloc,
                'feat2w_corloc':    feat2w_corloc,
                'feat1c_corloc_org':feat1c_corloc_org,
                'feat2c_corloc_org':feat2_c_corloc_n_org,
                'feat1g_std':       feat1g_std, 
                'feat2g_std':       feat2g_std,
                'feat1w_std':       feat1w_std, 
                'feat2w_std':       feat2w_std,
                'temperature':      temperature,
                'valid_epi1':       valid1, 
                'valid_epi2':       valid2
                }

class Preprocess_Skip(nn.Module):
    '''
    the preprocess class for keypoint detection net training
    '''
    def __init__(self, **kargs):
        super(Preprocess_Skip, self).__init__()
        self.__lossname__ = 'Preprocess_Skip'

    def forward(self, inputs, outputs):
        return None