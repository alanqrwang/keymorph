import torch
import torch.nn as nn


class CenterOfMass2d(nn.Module):
    def __init__(self):
        super(CenterOfMass2d, self).__init__()

    def forward(self, img):
        """
        x: tensor of shape [n_batch, chs, dimy, dimx]
        returns: center of mass in normalized coordinates [0,1]x[0,1], shape [n_batch, chs, 2]
        """
        n_batch, chs, dimy, dimx = img.shape
        eps = 1e-8
        arangex = (
            torch.linspace(0, 1, dimx).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )
        arangey = (
            torch.linspace(0, 1, dimy).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )

        arangex, arangey = arangex.to(img.device), arangey.to(img.device)

        mx = img.sum(dim=-2)  # mass along the dimN, shape [n_batch, chs, dimN]
        Mx = mx.sum(dim=-1, keepdim=True) + eps  # total mass along dimN

        my = img.sum(dim=-1)
        My = my.sum(dim=-1, keepdim=True) + eps

        # center of mass along dimN, shape [n_batch, chs, 1]
        cx = (arangex * mx).sum(dim=-1, keepdim=True) / Mx
        cy = (arangey * my).sum(dim=-1, keepdim=True) / My

        # center of mass, shape [n_batch, chs, 2]
        return torch.cat([cx, cy], dim=-1)


class CenterOfMass3d(nn.Module):
    def __init__(self):
        super(CenterOfMass3d, self).__init__()

    def forward(self, vol):
        """
        x: tensor of shape [n_batch, chs, dimz, dimy, dimx]
        returns: center of mass in normalized coordinates [0,1]x[0,1]x[0,1], shape [n_batch, chs, 2]
        """
        n_batch, chs, dimz, dimy, dimx = vol.shape
        eps = 1e-8
        arangex = (
            torch.linspace(0, 1, dimx).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )
        arangey = (
            torch.linspace(0, 1, dimy).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )
        arangez = (
            torch.linspace(0, 1, dimz).float().view(1, 1, -1).repeat(n_batch, chs, 1)
        )

        arangex, arangey, arangez = (
            arangex.to(vol.device),
            arangey.to(vol.device),
            arangez.to(vol.device),
        )

        mx = vol.sum(dim=(2, 3))  # mass along the dimN, shape [n_batch, chs, dimN]
        Mx = mx.sum(dim=-1, keepdim=True) + eps  # total mass along dimN

        my = vol.sum(dim=(2, 4))
        My = my.sum(dim=-1, keepdim=True) + eps

        mz = vol.sum(dim=(3, 4))
        Mz = mz.sum(dim=-1, keepdim=True) + eps

        # center of mass along dimN, shape [n_batch, chs, 1]
        cx = (arangex * mx).sum(dim=-1, keepdim=True) / Mx
        cy = (arangey * my).sum(dim=-1, keepdim=True) / My
        cz = (arangez * mz).sum(dim=-1, keepdim=True) / Mz

        # center of mass, shape [n_batch, chs, 3]
        return torch.cat([cx, cy, cz], dim=-1)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, norm_type, down_sample=True, dim=2
    ):
        super(ConvBlock, self).__init__()
        self.norm_type = norm_type
        self.down_sample = down_sample

        if dim == 2:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
            self.down = nn.MaxPool2d(2)

        elif dim == 3:
            if norm_type == "none":
                self.norm = None
            elif norm_type == "instance":
                self.norm = nn.InstanceNorm3d(out_channels)
            elif norm_type == "batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm_type == "group":
                self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
            else:
                raise NotImplementedError()

            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
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
