import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseNorm2(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon    = epsilon
        self.conv1      = nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3)
        self.conv2      = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
    def forward(self,x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(self.epsilon).sqrt()
        output = (x - mean) / std
        map = torch.mean(x,dim=1, keepdim=True)
        map1 = self.conv1(map)
        map2 = self.conv2(map)
        return output*map1 + map2

class Adaffusion(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels   = num_channels
        self.avg_pool       = nn.AdaptiveAvgPool2d(1)
        self.fc1            = nn.Linear(128,64)
        self.relu1          = nn.ReLU()
        self.fc2            = nn.Linear(64,128)
        self.fc3            = nn.Linear(128, 64)
        self.relu2          = nn.ReLU()
        self.fc4            = nn.Linear(64, 128)

    def forward(self, result, x):
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        avg_out2 = self.fc4(self.relu2(self.fc3(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        return result * avg_out1 + avg_out2

class ChannelwiseNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affusion=False, track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        if affusion:
            self.affusion = Adaffusion(num_features)
        else:
            self.affusion = None

        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape

        if self.training or not self.track_running_stats:
            # All dims except for B and C
            mu = x.mean(dim=(2, 3))
            sigma = x.var(dim=(2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var
            b = 1

        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * ((h * w) / ((h * w) - 1))
            self.running_mean   = self.running_mean * (1 - self.momentum) + mu.mean(dim=0) * self.momentum
            self.running_var    = self.running_var * (1 - self.momentum) + sigma_unbiased.mean(dim=0) * self.momentum

        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affusion is not None:
            result = self.affusion(result)

        return result

class DetNet(nn.Module):
    """
    Detection Head: Detection network with Peakiness Measurements
    """
    def __init__(self, descnet, in_channels, out_channels=1):
        super(DetNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels+64, 128, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm2d(out_channels)
        self.relu  = nn.PReLU()

        self.convimg = nn.Conv2d(3, 64, 3, 1, 1)
        self.normimg = nn.InstanceNorm2d(64)

    def name(self):
        return 'DetNet'

    def peakiness_score(self, x, ksize=3, dilation=1):
        '''
        compute the peakiness score map
        :param x: [b, c, h, w] the feature maps
        :return: [b, 1, h, w] the peakiness score map
        '''
        b,c,h,w = x.shape
        max_per_sample = torch.max(x.view(b,-1), dim=1)[0]
        x = x / max_per_sample.view(b,1,1,1)

        pad_inputs = F.pad(x, [dilation]*4, mode='reflect')
        avg_inputs = F.avg_pool2d(pad_inputs, ksize, stride=1)
        
        alpha   = F.softplus(x - avg_inputs)
        beta    = F.softplus(x - x.mean(1, True))

        score_vol = alpha * beta
        score_map = score_vol.max(1,True)[0]

        return score_map

    def forward(self, fusion_maps, train_desc=False):
        assert len(fusion_maps) == 2
        
        fusion_map  = fusion_maps[0]
        img_tensor  = fusion_maps[1]

        x_pf = self.peakiness_score(fusion_map)
        x_pi = self.peakiness_score(img_tensor)
        
        x           = self.relu(self.norm1(self.conv1(x_pf*fusion_map)))
        x           = F.interpolate(x, img_tensor.shape[2:], align_corners=False, mode='bilinear')
        
        img_tensor  = self.relu(self.normimg(self.convimg(x_pi*img_tensor)))

        x = torch.cat([x, img_tensor], dim=1)
        x = self.relu(self.norm2(self.conv2(x)))

        score = F.softplus(self.norm3(self.conv3(x)))

        score = F.interpolate(x_pf, img_tensor.shape[2:], align_corners=False, mode='bilinear') * x_pi * score

        return score
