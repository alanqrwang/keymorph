import torch
import torch.nn.functional as F
import scipy
from scipy.ndimage import morphology
import numpy as np

class MSELoss(torch.nn.Module):
  '''MSE loss.'''
  def forward(self, pred, target):
    return F.mse_loss(pred, target)

class DiceLoss(torch.nn.Module):
  '''Dice loss (lower is better).

  Supports 2d or 3d inputs.
  Supports hard dice or soft dice.
  If soft dice, returns scalar dice loss for entire slice/volume.
  If hard dice, returns:
      total_avg: scalar dice loss for entire slice/volume ()
      regions_avg: dice loss per region (n_ch)
      ind_avg: dice loss per region per pixel (bs, n_ch)

  '''
  def __init__(self, hard=False):
    super(DiceLoss, self).__init__()
    self.hard = hard

  def forward(self, pred, target, ign_first_ch=False):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'
    
    if len(target.size())==4:
      n,c,_,_ = target.size()
    if len(target.size())==5:
      n,c,_,_,_ = target.size()

    target = target.contiguous().view(n,c,-1)
    pred = pred.contiguous().view(n,c,-1)
    
    if self.hard: # hard Dice
      pred_onehot = torch.zeros_like(pred)
      pred = torch.argmax(pred, dim=1, keepdim=True)
      pred = torch.scatter(pred_onehot, 1, pred, 1.)
    if ign_first_ch:
      target = target[:,1:,:]
      pred = pred[:,1:,:]
  
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss
    total_avg = torch.mean(dice_loss)
    regions_avg = torch.mean(dice_loss, 0)
    
    if not self.hard:
      return total_avg
    else:
      return total_avg, regions_avg, ind_avg

def _check_type(t):
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
    return t

# Hausdorff Distance
def _surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = (input_1.astype(int) - morphology.binary_erosion(input_1, conn).astype(int)).astype(bool)
    Sprime = (input_2.astype(int) - morphology.binary_erosion(input_2, conn).astype(int)).astype(bool)
    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
        
    return sds

def hausdorff_distance(test_seg, gt_seg):
    '''Computes Hausdorff distance on brain surface.

    Assumes segmentation maps are one-hot and first channel is background.
    
    Args:
        test_seg: Test segmentation map (bs, n_ch, l, w, h)
        gt_seg: Ground truth segmentation map (bs, n_ch, l, w, h)
    '''
    test_seg = _check_type(test_seg)
    gt_seg = _check_type(gt_seg)

    hd = 0
    for i in range(len(test_seg)):
        hd += _surfd(test_seg[i, 0], gt_seg[i, 0], [1.25, 1.25, 10], 1).max()
    return hd / len(test_seg)

# Jacobian determinant
def _jacobian_determinant(disp):
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grad_disp = np.concatenate([gradz_disp, grady_disp, gradx_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def jdstd(disp):
    disp = _check_type(disp)
    jd = _jacobian_determinant(disp)
    return jd.std()

def jdlessthan0(disp, as_percentage=False):
    disp = _check_type(disp)
    jd = _jacobian_determinant(disp)
    if as_percentage:
        return np.count_nonzero(jd <= 0) / len(jd.flatten())
    return np.count_nonzero(jd <= 0)