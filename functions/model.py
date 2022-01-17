import torch.nn as nn
from .registration_tools import center_of_mass

class KPmorph(nn.Module):
    """
    Create dataloader
    
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
        super(KPmorph, self).__init__()
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

        if norm_type==0:
            self.norm = None
        elif norm_type==1:
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm_type==2:
            self.norm = nn.BatchNorm3d(out_channels)
        else:
            raise Exception('Choose between 1:Instance 2:Batch')

        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride= stride,
                              padding=1)
        self.activation = nn.ReLU(out_channels)
        
        self.down = nn.Conv3d(out_channels,
                              out_channels,
                              kernel_size=2,
                              stride=2)

    def forward(self, x):
        out = self.conv(x)
        if self.norm_type>0:
            out = self.norm(out)
        out = self.activation(out)
        if self.down_sample:
            out = self.down(out)
        return out