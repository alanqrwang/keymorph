import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from skimage import morphology
from .registration_tools import center_of_mass
from . import layers

class ConvNetFC(nn.Module):
  def __init__(self, dim, input_ch, h_dims, out_dim, norm_type):
    super(ConvNetFC, self).__init__()
    self.dim = dim

    self.block1 = layers.ConvBlock(input_ch, h_dims[0], 1, norm_type, False, dim)
    self.block2 = layers.ConvBlock(h_dims[0], h_dims[1], 1, norm_type, False, dim)

    self.block3 = layers.ConvBlock(h_dims[1], h_dims[2], 1, norm_type, False, dim)
    self.block4 = layers.ConvBlock(h_dims[2], h_dims[3], 1, norm_type, False, dim)

    self.block5 = layers.ConvBlock(h_dims[3], h_dims[4], 1, norm_type, False, dim)
    self.block6 = layers.ConvBlock(h_dims[4], h_dims[5], 1, norm_type, False, dim)

    self.block7 = layers.ConvBlock(h_dims[5], h_dims[6], 1, norm_type, False, dim)
    self.block8 = layers.ConvBlock(h_dims[6], h_dims[7], 1, norm_type, False, dim)

    self.fc = nn.Linear(h_dims[7], out_dim)

  def forward(self, x):

    out = self.block1(x)
    out = self.block2(out)
    out = self.block3(out)
    out = self.block4(out)
    out = self.block5(out)
    out = self.block6(out)
    out = self.block7(out)
    out = self.block8(out)

    if self.dim == 2:
      out = F.avg_pool2d(out,
                kernel_size=out.size()[-1]).view(out.size()[0],-1)
    elif self.dim == 3:
      out = F.avg_pool3d(out,
                kernel_size=out.size()[-1]).view(out.size()[0],-1)

    out = self.fc(out)
    out = torch.sigmoid(out)
    return out*2-1 #[-1, 1] for F.grid_sample

class ConvNetCoM(nn.Module):
  def __init__(self, dim, input_ch, h_dims, out_dim, norm_type):
    super(ConvNetCoM, self).__init__()
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
    if self.dim == 2:
      self.com = layers.CenterOfMass2d()
    elif self.dim == 3:
      self.com = layers.CenterOfMass3d()
    self.relu = nn.ReLU()

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
    out = self.relu(out)
    out = self.com(out)
    return out*2-1 #[-1, 1] for F.grid_sample

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
