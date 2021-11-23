import torch.nn as nn
import torch.nn.functional as F
import torch

# adapt from https://github.com/MIC-DKFZ/BraTS2017


def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=8):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor

class InitConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, dropout=0.2):
        super(InitConv, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = dropout

    def forward(self, x):
        y = self.conv(x)
        y = F.dropout3d(y, self.dropout)

        return y


class EnBlock(nn.Module):
    def __init__(self, in_channels, norm='gn'):
        super(EnBlock, self).__init__()

        self.bn1 = normalization(in_channels, norm=norm)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

        self.bn2 = normalization(in_channels, norm=norm)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        y = self.bn2(x1)
        y = self.relu2(y)
        y = self.conv2(y)
        y = y + x

        return y


class EnDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnDown, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = self.conv(x)

        return y



class Unet(nn.Module):
    def __init__(self, in_channels=1, base_channels=8, num_classes=3):
        super(Unet, self).__init__()

        self.InitConv = InitConv(in_channels=in_channels, out_channels=8, dropout=0.2)

        # I added!
        self.EnBlock0_1 = EnBlock(in_channels=8)
        self.EnBlock0_2 = EnBlock(in_channels=8)
        self.SE0 = ChannelSELayer3D(num_channels=8)
        self.EnDown0 = EnDown(in_channels=8, out_channels=16)

        self.EnBlock1_1 = EnBlock(in_channels=16)
        self.EnBlock1_2 = EnBlock(in_channels=16)
        self.SE1 = ChannelSELayer3D(num_channels=16)
        self.EnDown1 = EnDown(in_channels=16, out_channels=32)

        self.EnBlock2_1 = EnBlock(in_channels=32)
        self.EnBlock2_2 = EnBlock(in_channels=32)
        self.SE2 = ChannelSELayer3D(num_channels=32)
        self.EnDown2 = EnDown(in_channels=32, out_channels=64)

        self.EnBlock3_1 = EnBlock(in_channels=64)
        self.EnBlock3_2 = EnBlock(in_channels=64)
        self.SE3 = ChannelSELayer3D(num_channels=64)
        self.EnDown3 = EnDown(in_channels=64, out_channels=128)

        self.EnBlock4_1 = EnBlock(in_channels=128)
        self.EnBlock4_2 = EnBlock(in_channels=128)
        self.EnBlock4_3 = EnBlock(in_channels=128)
        self.EnBlock4_4 = EnBlock(in_channels=128)
        # self.SE4 = ChannelSELayer3D(in_channels=128)

    def forward(self, x):

        x = self.InitConv(x)       # (1, 8, 192, 192, 192)

        # I added this extra encoding operation
        x0_1 = self.EnBlock0_1(x)
        x0_1 = self.EnBlock0_2(x0_1)
        x0_1 = self.SE0(x0_1)
        x0_2 = self.EnDown0(x0_1)  # (1, 16, 96, 96, 96)

        x1_1 = self.EnBlock1_1(x0_2)
        x1_1 = self.EnBlock1_2(x1_1)
        x1_1 = self.SE1(x1_1)
        x1_2 = self.EnDown1(x1_1)  # (1, 32, 48, 48, 48)

        x2_1 = self.EnBlock2_1(x1_2)
        x2_1 = self.EnBlock2_2(x2_1)
        x2_1 = self.SE2(x2_1)
        x2_2 = self.EnDown2(x2_1)  # (1, 64, 24, 24, 24)

        x3_1 = self.EnBlock3_1(x2_2)
        x3_1 = self.EnBlock3_2(x3_1)
        x3_1 = self.SE3(x3_1)
        x3_2 = self.EnDown3(x3_1)  # (1, 128, 12, 12, 12)

        x4_1 = self.EnBlock4_1(x3_2)
        x4_2 = self.EnBlock4_2(x4_1)
        x4_3 = self.EnBlock4_3(x4_2)

        output = self.EnBlock4_4(x4_3)  # (1, 128, 12, 12, 12)

        return x0_1,x1_1,x2_1,x3_1,output

# Not really relevant
if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        model.cuda()
        output = model(x)
        print('output:', output.shape)
