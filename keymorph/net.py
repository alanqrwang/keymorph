import torch.nn as nn
from . import layers

h_dims = [32, 64, 64, 128, 128, 256, 256, 512]


class ConvNet(nn.Module):
    def __init__(self, dim, input_ch, out_dim, norm_type):
        super(ConvNet, self).__init__()
        self.dim = dim

        self.block1 = layers.ConvBlock(input_ch, h_dims[0], 1, norm_type, False, dim)
        self.block2 = layers.ConvBlock(h_dims[0], h_dims[1], 1, norm_type, True, dim)

        self.block3 = layers.ConvBlock(h_dims[1], h_dims[2], 1, norm_type, False, dim)
        self.block4 = layers.ConvBlock(h_dims[2], h_dims[3], 1, norm_type, True, dim)

        self.block5 = layers.ConvBlock(h_dims[3], h_dims[4], 1, norm_type, False, dim)
        self.block6 = layers.ConvBlock(h_dims[4], h_dims[5], 1, norm_type, True, dim)

        self.block7 = layers.ConvBlock(h_dims[5], h_dims[6], 1, norm_type, False, dim)
        self.block8 = layers.ConvBlock(h_dims[6], h_dims[7], 1, norm_type, True, dim)

        self.block9 = layers.ConvBlock(h_dims[7], out_dim, 1, norm_type, False, dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        return out
