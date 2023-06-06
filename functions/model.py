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

