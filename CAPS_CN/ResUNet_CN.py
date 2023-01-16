import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        # self.bn = nn.BatchNorm2d(num_out_layers) # 
        self.gn = nn.GroupNorm(num_groups=32,num_channels=num_out_layers)

    def forward(self, x):
        return F.elu(self.gn(self.conv(x)), inplace=True)

class pure_conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(pure_conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)

    def forward(self, x):
        return self.conv(x)

class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)

class PositionwiseNorm2(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.conv1=nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
    def forward(self,x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(self.epsilon).sqrt()
        output = (x - mean) / std
        map = torch.mean(x,dim=1, keepdim=True)
        map1=self.conv1(map)
        map2=self.conv2(map)
        return output*map1+map2

class Adaffine(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(128,64)
        self.relu1 = nn.ReLU()
        self.fc2=nn.Linear(64,128)
        self.fc3 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(64, 128)

    def forward(self, result, x):
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        avg_out2 = self.fc4(self.relu2(self.fc3(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        return result * avg_out1 + avg_out2

class ChannelwiseNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affine=False, track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        if affine:
            self.affine = Adaffine(num_features)
        else:
            self.affine = None

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
            self.running_mean = self.running_mean * (1 - self.momentum) + mu.mean(dim=0) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + sigma_unbiased.mean(dim=0) * self.momentum

        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affine is not None:
            result = self.affine(result)

        return result

class Upsample(nn.Module):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)

def upsample_like(src,tar):

    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')

    return src

class ResUNetCN(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True,
                 coarse_out_ch=128,
                 fine_out_ch=128
                 ):

        super(ResUNetCN, self).__init__()
        # assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        # if encoder in ['resnet18', 'resnet34']:
        #     filters = [64, 128, 256, 512]
        # else:
        #     filters = [256, 512, 1024, 2048]

        resnet = class_for_name("torchvision.models", 'resnet50')(pretrained=pretrained)

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

        self.firstmaxpool = resnet.maxpool  # H/4
        
        # # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16
        del resnet

        # coarse-level conv
        self.conv_coarse = conv(1024, coarse_out_ch, 1, 1)

        # cross norm
        self.coarse_pn = PositionwiseNorm2()
        self.coarse_cn = ChannelwiseNorm(coarse_out_ch)
        self.fuse_weight_coarse_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_coarse_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_coarse_1.data.fill_(0.7)
        self.fuse_weight_coarse_2.data.fill_(0.3)

        # decoder
        self.upconv3 = upconv(1024, 512, 3, 2)
        self.iconv3 = conv(512 + 512, 512, 3, 1)
        self.upconv2 = upconv(512, 256, 3, 2)
        self.iconv2 = conv(256 + 256, 256, 3, 1)

        # fine-level conv
        self.conv_fine = conv(256, fine_out_ch, 1, 1)

        # cross norm
        self.fine_pn = PositionwiseNorm2()
        self.fine_cn = ChannelwiseNorm(fine_out_ch)
        self.fuse_weight_fine_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_fine_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_fine_1.data.fill_(0.7)
        self.fuse_weight_fine_2.data.fill_(0.3)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):

        # Encoder
        x_first = self.firstrelu(self.firstbn(self.firstconv(x)))
        x_first_pool = self.firstmaxpool(x_first)

        x1 = self.layer1(x_first_pool)
        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        # Coarse
        x_coarse = self.conv_coarse(x3)
        desc1 = self.coarse_pn(x_coarse)
        desc2 = self.coarse_cn(x_coarse)
        x_coarse_cn = desc1 * (self.fuse_weight_coarse_1/(self.fuse_weight_coarse_1+self.fuse_weight_coarse_2)) + \
             desc2 * (self.fuse_weight_coarse_2/(self.fuse_weight_coarse_1+self.fuse_weight_coarse_2))
        
        # Decoder
        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        d2 = self.iconv3(x)

        x = self.upconv2(d2)
        x = self.skipconnect(x1, x)
        d1 = self.iconv2(x)

        # Fine
        x_fine = self.conv_fine(d1)
        desc1 = self.fine_pn(x_fine)
        desc2 = self.fine_cn(x_fine)
        x_fine_cn = desc1 * (self.fuse_weight_fine_1/(self.fuse_weight_fine_1+self.fuse_weight_fine_2)) + \
             desc2 * (self.fuse_weight_fine_2/(self.fuse_weight_fine_1+self.fuse_weight_fine_2))

        return {'desc_map': [x_coarse, x_fine], 'local_map': [x_coarse_cn, x_fine_cn]}