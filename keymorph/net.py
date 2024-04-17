import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet3d.model import UNet2D, UNet3D, TruncatedUNet3D
from . import layers

from se3cnn.image.gated_block import GatedBlock

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


class UNet(nn.Module):
    def __init__(self, dim, input_ch, out_dim):
        super(UNet, self).__init__()
        if dim == 2:
            backbone = UNet2D(
                input_ch,
                out_dim,
                final_sigmoid=False,
                f_maps=64,
                layer_order="gcr",
                num_groups=8,
                num_levels=4,
                is_segmentation=False,
                conv_padding=1,
            )
        if dim == 3:
            backbone = UNet3D(
                input_ch,
                out_dim,
                final_sigmoid=False,
                f_maps=32,  # Used by nnUNet
                layer_order="gcr",
                num_groups=8,
                num_levels=4,
                is_segmentation=False,
                conv_padding=1,
            )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class RXFM_Net(nn.Module):
    def __init__(self, n_in, output_chans, norm_type):
        super(RXFM_Net, self).__init__()

        chan_config = [[16, 16, 4], [16, 16, 4], [16, 16, 4], [16, 16, 4]]
        features = [[n_in]] + chan_config + [[output_chans]]

        common_block_params = {
            "size": 5,
            "stride": 2,
            "padding": 2,
            "normalization": norm_type,
            "capsule_dropout_p": None,
            "smooth_stride": False,
        }

        block_params = [{"activation": F.relu}] * (len(features) - 2) + [
            {"activation": F.relu}
        ]

        assert len(block_params) + 1 == len(features)

        blocks = [
            GatedBlock(
                features[i],
                features[i + 1],
                **common_block_params,
                **block_params[i],
                dyn_iso=True
            )
            for i in range(len(block_params))
        ]

        self.sequence = torch.nn.Sequential(*blocks)

    def forward(self, x):
        x = self.sequence(x)
        return x


class TruncatedUNet(nn.Module):
    def __init__(self, dim, input_ch, out_dim, num_truncated_layers):
        super(TruncatedUNet, self).__init__()
        assert dim == 3
        backbone = TruncatedUNet3D(
            input_ch,
            out_dim,
            num_truncated_layers,
            final_sigmoid=False,
            f_maps=32,  # Used by nnUNet
            layer_order="gcr",
            num_groups=8,
            num_levels=4,
            is_segmentation=False,
            conv_padding=1,
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
