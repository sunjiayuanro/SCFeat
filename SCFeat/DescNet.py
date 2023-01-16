import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)

class ResUNet_F2R(nn.Module):
    """
    F2R-Backbone: Feature-Fusion-ResUNet Backbone
    """
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True,
                 fusion_out_ch=128
                 ):

        super(ResUNet_F2R, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        self.firstconv      = resnet.conv1  # H/2
        self.firstbn        = resnet.bn1
        self.firstrelu      = resnet.relu
        self.firstmaxpool   = resnet.maxpool  # H/4

        # Encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

        # Decoder
        self.upconv3    = upconv(filters[2], 512, 3, 2)
        self.iconv3     = conv(filters[1] + 512, 512, 3, 1)
        self.upconv2    = upconv(512, 256, 3, 2)
        self.iconv2     = conv(filters[0] + 256, 256, 3, 1)

	    # Feature Fusion Block
        self.side3          = upconv_like(1024, fusion_out_ch, 3)
        self.side2          = upconv_like(512, fusion_out_ch, 3)
        self.side1          = conv(256, fusion_out_ch, 1, 1)
        self.fusion_conv    = nn.Conv2d(3*fusion_out_ch,fusion_out_ch,1)

        # Cross Norm. Layer
        self.fusion_pn = PositionwiseNorm2()
        self.fusion_cn = ChannelwiseNorm(fusion_out_ch)
        self.fuse_weight_fusion_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_fusion_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_fusion_1.data.fill_(0.7)
        self.fuse_weight_fusion_2.data.fill_(0.3)

        self.out_channels = 192

    def name(self):
        return 'ResUNet_F2R'
        
    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):

        # Encoder
        x       = self.firstrelu(self.firstbn(self.firstconv(x)))
        x_first = self.firstmaxpool(x)
        
        x1 = self.layer1(x_first)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        # Decoder
        x   = self.upconv3(x3)
        x   = self.skipconnect(x2, x)
        x2d = self.iconv3(x)

        x = self.upconv2(x2d)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        # Feature Fusion output
        d1          = self.side1(x)
        d2          = self.side2(x2d, d1)
        d3          = self.side3(x3, d1)
        x_fusion    = self.fusion_conv(torch.cat((d1,d2,d3),1))  # H/4
        
        del x, x2, d1, d2, d3, x2d

        # Shared Coupling-bridge Normalization
        desc1       = self.fusion_pn(x_fusion)
        desc2       = self.fusion_cn(x_fusion)
        x_fusion_cn = desc1 * (self.fuse_weight_fusion_1/(self.fuse_weight_fusion_1+self.fuse_weight_fusion_2)) + \
            desc2 * (self.fuse_weight_fusion_2/(self.fuse_weight_fusion_1+self.fuse_weight_fusion_2))

        return {
            'global_map':       x3,             # Coarse level
            'desc_map':         x_fusion,       # D_desc
            'local_map':        x_fusion_cn,    # F_cn
            'local_map_first':  x_first         # Initial feature maps
            }

class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.gn = nn.GroupNorm(num_groups=32,num_channels=num_out_layers) # Group Norm with 32

    def forward(self, x):
        return F.elu(self.gn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale  = scale
        self.conv   = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)

class upconv_like(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(upconv_like, self).__init__()

        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x, target):
        x = F.upsample(x,size=target.shape[2:],mode='bilinear')
        return self.conv(x)

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
