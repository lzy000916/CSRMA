import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchsummary.torchsummary import summary
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x



class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super(Basic_blocks, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2 + x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SECA(nn.Module):
    def __init__(self, in_channel, decay=2, b=1):
        super(SECA, self).__init__()
        t = int(abs((math.log(in_channel, 2) + b) / decay))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gapool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.conv1(gp.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        se = self.sigmoid(se)
        x = x * se
        gap = self.gapool(x)
        se2 = self.layer2(gap)
        return x * se2

# class DSE(nn.Module):
#     def __init__(self, in_channel, decay=2):
#         super(DSE, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel // decay, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // decay, in_channel, 1),
#             nn.Sigmoid()
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel // decay, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channel // decay, in_channel, 1),
#             nn.Sigmoid()
#         )
#         self.gpool = nn.AdaptiveAvgPool2d(1)
#         self.gapool = nn.AdaptiveMaxPool2d(1)
#
#     def forward(self, x):
#         gp = self.gpool(x)
#         se = self.layer1(gp)
#         x = x * se
#         gap = self.gapool(x)
#         se2 = self.layer2(gap)
#         return x * se2


class Spaceatt(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, in_channel // decay, 3, padding=1),
            ScConv(in_channel // decay),
            SELayer(in_channel // decay)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel // decay),
            ScConv(in_channel // decay),
            SELayer(in_channel // decay)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att = att @ V
        return self.sig(att)


class CSCA_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=2):
        super(CSCA_blocks, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(in_channel, out_channel // 2)
        self.catt = SELayer(out_channel // 2, decay)
        self.satt = Spaceatt(out_channel // 2, decay)

    def forward(self, high, low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt(point, catt)
        plusatt = catt * satt
        return torch.cat([plusatt, catt], dim=1)




class CSCAUNet(nn.Module):
    def __init__(self, n_class=1, decay=2):
        super(CSCAUNet, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.down_conv1 = Basic_blocks(3, 32, decay)
        self.down_conv2 = Basic_blocks(32, 64, decay)
        self.down_conv3 = Basic_blocks(64, 128, decay)
        self.down_conv4 = Basic_blocks(128, 256, decay)
        self.down_conv5 = Basic_blocks(256, 512, decay)


        self.down_conv6 = nn.Sequential(
            Basic_blocks(512, 1024, decay),
            SELayer(1024, decay)
        )

        self.up_conv5 = CSCA_blocks(1024, 512, decay)
        self.up_conv4 = CSCA_blocks(512, 256, decay)
        self.up_conv3 = CSCA_blocks(256, 128, decay)
        self.up_conv2 = CSCA_blocks(128, 64, decay)
        self.up_conv1 = CSCA_blocks(64, 32, decay)

        self.dp6 = nn.Conv2d(1024, 1, 1)
        self.dp5 = nn.Conv2d(512, 1, 1)
        self.dp4 = nn.Conv2d(256, 1, 1)
        self.dp3 = nn.Conv2d(128, 1, 1)
        self.dp2 = nn.Conv2d(64, 1, 1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)  # 103

        self.center5 = nn.Conv2d(1024, 512, 1)
        self.decodeup4 = nn.Conv2d(512, 256, 1)
        self.decodeup3 = nn.Conv2d(256, 128, 1)
        self.decodeup2 = nn.Conv2d(128, 64, 1)

    def forward(self, inputs):
        b, c, h, w = inputs.size()
        down1 = self.down_conv1(inputs)
        pool1 = self.pool(down1)

        down2 = self.down_conv2(pool1)
        pool2 = self.pool(down2)

        down3 = self.down_conv3(pool2)
        pool3 = self.pool(down3)

        down4 = self.down_conv4(pool3)
        pool4 = self.pool(down4)

        down5 = self.down_conv5(pool4)
        pool5 = self.pool(down5)

        center = self.down_conv6(pool5)

        out6 = self.dp6(center)
        out6 = F.interpolate(
            out6, (h, w), mode='bilinear', align_corners=False)

        deco5 = self.up_conv5(center, down5)
        out5 = self.dp5(deco5)
        out5 = F.interpolate(
            out5, (h, w), mode='bilinear', align_corners=False)
        center5 = self.center5(center)
        center5 = F.interpolate(center5, (h // 16, w // 16),
                                mode='bilinear', align_corners=False)
        deco5 = deco5 + center5

        deco4 = self.up_conv4(deco5, down4)
        out4 = self.dp4(deco4)
        out4 = F.interpolate(
            out4, (h, w), mode='bilinear', align_corners=False)
        decoderup4 = self.decodeup4(deco5)
        decoderup4 = F.interpolate(
            decoderup4, (h // 8, w // 8), mode='bilinear', align_corners=False)
        deco4 = deco4 + decoderup4

        deco3 = self.up_conv3(deco4, down3)
        out3 = self.dp3(deco3)
        out3 = F.interpolate(
            out3, (h, w), mode='bilinear', align_corners=False)
        decoderup3 = self.decodeup3(deco4)
        decoderup3 = F.interpolate(
            decoderup3, (h // 4, w // 4), mode='bilinear', align_corners=False)
        deco3 = deco3 + decoderup3

        deco2 = self.up_conv2(deco3, down2)
        out2 = self.dp2(deco2)
        out2 = F.interpolate(out2, (h, w), mode='bilinear', align_corners=False)
        decoderup2 = self.decodeup2(deco3)
        decoderup2 = F.interpolate(
            decoderup2, (h // 2, w // 2), mode='bilinear', align_corners=False
        )
        deco2 = deco2 + decoderup2

        deco1 = self.up_conv1(deco2, down1)
        out = self.out(deco1)
        return out, out2, out3, out4, out5, out6


if __name__ == '__main__':
    model = CSCAUNet(1, 2)
    summary(model, (3, 352, 352), batch_size=1, device='cpu')
    print('# generator parameters:', sum(param.numel()
       for param in model.parameters()))
