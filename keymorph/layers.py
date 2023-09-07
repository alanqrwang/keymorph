import torch
import torch.nn as nn

class CenterOfMass2d(nn.Module):
    def __init__(self):
        super(CenterOfMass2d, self).__init__()

    def forward(self, img):
        """
        x: tensor of shape [n_batch, chs, dim1, dim2]
        returns: center of mass, shape [n_batch, chs, 2]
        """
        n_batch, chs, dim1, dim2 = img.shape
        eps = 1e-8
        arangex = torch.linspace(0,1,dim2).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arangey = torch.linspace(0,1,dim1).float().view(1,1,-1).repeat(n_batch, chs, 1)
        
        arangex, arangey = arangex.to(img.device), arangey.to(img.device)
            
        mx = img.sum(dim=2) #mass along the dimN, shape [n_batch, chs, dimN] 
        Mx = mx.sum(-1, True) + eps #total mass along dimN

        my = img.sum(dim=3)
        My = my.sum(-1, True) + eps

        cx = (arangex*mx).sum(-1,True)/Mx #center of mass along dimN, shape [n_batch, chs, 1]
        cy = (arangey*my).sum(-1,True)/My

        C = torch.cat([cx,cy],-1) #center of mass, shape [n_batch, chs, 2]
        return C

class CenterOfMass3d(nn.Module):
    def __init__(self):
        super(CenterOfMass3d, self).__init__()

    def forward(self, vol):
        """
        x: tensor of shape [n_batch, chs, dim1, dim2, dim3]
        returns: center of mass, shape [n_batch, chs, 3]
        """
        n_batch, chs, dim1, dim2, dim3 = vol.shape
        eps = 1e-8
        arange1 = torch.linspace(0,1,dim1).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arange2 = torch.linspace(0,1,dim2).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arange3 = torch.linspace(0,1,dim3).float().view(1,1,-1).repeat(n_batch, chs, 1)
        
        arange1, arange2, arange3 = arange1.to(vol.device), arange2.to(vol.device), arange3.to(vol.device)
            
        mx = vol.sum(dim=(2,3)) #mass along the dimN, shape [n_batch, chs, dimN] 
        Mx = mx.sum(-1, True) + eps #total mass along dimN

        my = vol.sum(dim=(2,4))
        My = my.sum(-1, True) + eps

        mz = vol.sum(dim=(3,4))
        Mz = mz.sum(-1, True) + eps

        cx = (arange1*mx).sum(-1,True)/Mx #center of mass along dimN, shape [n_batch, chs, 1]
        cy = (arange2*my).sum(-1,True)/My
        cz = (arange3*mz).sum(-1,True)/Mz

        C = torch.cat([cx,cy,cz],-1) #center of mass, shape [n_batch, chs, 3]
        return C

class ConvBlock(nn.Module):
    def __init__(self,
          in_channels,
          out_channels,
          stride,
          norm_type,
          down_sample=True,
          dim=2):
        super(ConvBlock, self).__init__()
        self.norm_type = norm_type
        self.down_sample = down_sample

        if dim == 2:
            if norm_type == 'none':
                self.norm = None
            elif norm_type == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'group':
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()
            self.conv = nn.Conv2d(in_channels,
                        out_channels,
                        kernel_size=3,
                        stride= stride,
                        padding=1)
            self.down = nn.MaxPool2d(2)

        elif dim == 3:
            if norm_type == 'none':
                self.norm = None
            elif norm_type == 'instance':
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm_type == 'batch':
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm_type == 'group':
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()

            self.conv = nn.Conv3d(in_channels,
                        out_channels,
                        kernel_size=3,
                        stride= stride,
                        padding=1)
            self.down = nn.MaxPool3d(2)

        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm(out)
        out = self.activation(out)
        if self.down_sample:
            out = self.down(out)
        return out