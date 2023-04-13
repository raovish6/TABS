import torch
import torch.nn as nn
import torch.nn.functional as F

from .Transformer import TransformerModel
from .PositionalEncoding import LearnedPositionalEncoding

class up_conv_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv_3D, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class resconv_block_3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(resconv_block_3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True),
            nn.Conv3d(ch_out, ch_out, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.GroupNorm(8, ch_out),
            nn.ReLU(inplace = True)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in, ch_out, kernel_size = 1, stride = 1, padding = 0)

    def forward(self,x):

        residual = self.Conv_1x1(x)
        x = self.conv(x)
        return residual + x

class TABS(nn.Module):
    def __init__(
        self,
        img_dim = 192,
        patch_dim = 8,
        img_ch = 1,
        output_ch = 1,
        embedding_dim = 512,
        num_heads = 8,
        num_layers = 4,
        dropout_rate = 0.1,
        attn_dropout_rate = 0.1,
        ):
        super(TABS,self).__init__()

        self.hidden_dim = int((img_dim/16)**3)

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Conv1 = resconv_block_3D(ch_in=img_ch,ch_out=8)
        self.Conv2 = resconv_block_3D(ch_in=8,ch_out=16)
        self.Conv3 = resconv_block_3D(ch_in=16,ch_out=32)
        self.Conv4 = resconv_block_3D(ch_in=32,ch_out=64)
        self.Conv5 = resconv_block_3D(ch_in=64,ch_out=128)

        self.Up5 = up_conv_3D(ch_in=128,ch_out=64)
        self.Up_conv5 = resconv_block_3D(ch_in=128, ch_out=64)
        self.Up4 = up_conv_3D(ch_in=64,ch_out=32)
        self.Up_conv4 = resconv_block_3D(ch_in=64, ch_out=32)
        self.Up3 = up_conv_3D(ch_in=32,ch_out=16)
        self.Up_conv3 = resconv_block_3D(ch_in=32, ch_out=16)
        self.Up2 = up_conv_3D(ch_in=16,ch_out=8)
        self.Up_conv2 = resconv_block_3D(ch_in=16, ch_out=8)

        self.Conv_1x1 = nn.Conv3d(8,output_ch,kernel_size=1,stride=1,padding=0)
        self.gn = nn.GroupNorm(8, 128)
        self.relu = nn.ReLU(inplace=True)
        self.act = nn.Softmax(dim=1)

        self.num_patches = int((img_dim // patch_dim) ** 3)

        self.position_encoding = LearnedPositionalEncoding(
            embedding_dim, self.hidden_dim
        )

        self.reshaped_conv = conv_block_3D(512, 128)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            self.hidden_dim,

            dropout_rate,
            attn_dropout_rate,
        )

        self.conv_x = nn.Conv3d(
            128,
            embedding_dim,
            kernel_size=3,
            stride=1,
            padding=1
            )

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.embedding_dim = embedding_dim

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x = self.Conv5(x5)

        x = self.gn(x)
        x = self.relu(x)
        x = self.conv_x(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(x.size(0), -1, self.embedding_dim)

        x = self.position_encoding(x)
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        encoder_outputs = {}
        all_keys = []
        for i in [1, 2, 3, 4]:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        x = encoder_outputs[all_keys[0]]
        x = self.reshape_output(x)
        x = self.reshaped_conv(x)

        d5 = self.Up5(x)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.act(d1)

        return d1

    def reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            int(self.img_dim//2 / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

if __name__ == '__main__':

    model = TABS()
    test = torch.rand([1,1,192,192,192])
    test = test.cuda(0)
    model = model.cuda(0)

    with torch.no_grad():
        out = model(test)
    print(out.shape)
