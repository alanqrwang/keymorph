import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from skimage import morphology
from .registration_tools import center_of_mass



"""KeyMorph Model"""


class KeyMorph(nn.Module):
    """
    Neural network for KeyMorph
    
    Arguments
    ---------
    input_ch   : input channel to the network
    out_dim    : output dimension of the network
    norm_type  : Type of layer norm. None:0, 1:Intstance, 2:Batch
    Return
    ------
        model : torch model 
    """

    def __init__(self, input_ch, out_dim, norm_type):
        super(KeyMorph, self).__init__()
        self.out_dim = out_dim

        self.block1 = cm_block(input_ch, 32, 1, norm_type, False)
        self.block2 = cm_block(32, 64, 1, norm_type)

        self.block3 = cm_block(64, 64, 1, norm_type, False)
        self.block4 = cm_block(64, 128, 1, norm_type)

        self.block5 = cm_block(128, 128, 1, norm_type, False)
        self.block6 = cm_block(128, 256, 1, norm_type)

        self.block7 = cm_block(256, 256, 1, norm_type, False)
        self.block8 = cm_block(256, 512, 1, norm_type, False)

        self.block9 = cm_block(512, out_dim, 1, norm_type, False)

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
        out = center_of_mass(out, True)

        return out


class cm_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 norm_type,
                 down_sample=True):

        super(cm_block, self).__init__()

        self.norm_type = norm_type
        self.down_sample = down_sample

        if norm_type == 0:
            self.norm = None
        elif norm_type == 1:
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_type == 2:
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            raise Exception('Choose between 1:Instance 2:Batch')

        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=stride,
                              padding=1)
        self.activation = nn.ReLU(out_channels)

        self.down = nn.Conv3d(out_channels,
                              out_channels,
                              kernel_size=2,
                              stride=2)

    def forward(self, x):
        out = self.conv(x)
        if self.norm_type > 0:
            out = self.norm(out)
        out = self.activation(out)
        if self.down_sample:
            out = self.down(out)
        return out


"""Brain Extractor Model"""


class Simple_Unet(nn.Module):   
    """
    Neural network for Brain Extractor
    
    Arguments
    ---------
    input_ch   : input channel to the network
    out_ch     : output dimension of the network
    use_in     : use instance norm
    enc_nf     : list of int for the encoder filter size
    dec_nf     : list of int for the decoder filter size
    Return
    ------
        model : torch model 
    """
    def __init__(self, input_ch, out_ch, use_in, enc_nf, dec_nf):
        super(Simple_Unet, self).__init__()

        self.down = torch.nn.MaxPool3d(2, 2)

        self.block0 = simple_block(input_ch, enc_nf[0], use_in)
        self.block1 = simple_block(enc_nf[0], enc_nf[1], use_in)
        self.block2 = simple_block(enc_nf[1], enc_nf[2], use_in)
        self.block3 = simple_block(enc_nf[2], enc_nf[3], use_in)

        self.block4 = simple_block(enc_nf[3], dec_nf[0], use_in)

        self.block5 = simple_block(dec_nf[0] * 2, dec_nf[1], use_in)
        self.block6 = simple_block(dec_nf[1] * 2, dec_nf[2], use_in)
        self.block7 = simple_block(dec_nf[2] * 2, dec_nf[3], use_in)
        self.block8 = simple_block(dec_nf[3] * 2, out_ch, use_in)

        self.conv = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x_in):
        # Model
        x0 = self.block0(x_in)
        x1 = self.block1(self.down(x0))
        x2 = self.block2(self.down(x1))
        x3 = self.block3(self.down(x2))

        x = self.block4(self.down(x3))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        x = torch.cat([x, x3], 1)
        x = self.block5(x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        x = torch.cat([x, x2], 1)
        x = self.block6(x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        x = torch.cat([x, x1], 1)
        x = self.block7(x)
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        x = torch.cat([x, x0], 1)
        x = self.block8(x)

        out = self.conv(x)

        return out


class simple_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_in):
        super(simple_block, self).__init__()

        self.use_in = use_in

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.InstanceNorm3d(out_channels)

        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_in:
            out = self.bn1(out)
        out = self.activation(out)
        return out


"""Helper function for brain extractor"""


def clean_mask(mask, threshold=0.2):
    """
    Remove small predicted segmentation. It finds the largest connected component.
    If there are other region/islands that is less than threshold percentage in size, 
    remove those region. 
    
    Arguments
    ---------
    mask         : numpy 3D binary mask for use for brain extraction
    threshold    : remove if not size(obj)/size(largest_component) > threshold
    
    Return
    ------
        new_mask : cleaned up mask with smaller regions removed
    """

    connected = morphology.label(mask)
    islands = np.unique(connected)[1:]

    islands_size = {}
    max_size = []
    for i in islands:
        size = (connected == i).sum()
        islands_size[i] = size
        max_size += [size]

    max_size = np.max(max_size)

    _island_size = islands_size.copy()
    for key in _island_size:
        if not islands_size[key] / max_size > threshold:
            islands_size.pop(key, None)

    new_mask = np.zeros_like(mask).astype('uint8')
    for key in islands_size:
        new_mask += (connected == key).astype('uint8')

    return new_mask
