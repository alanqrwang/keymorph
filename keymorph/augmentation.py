import numpy as np
import torch
import torch.nn.functional as F

class AffineDeformation2d():
  def __init__(self, device='cuda:0'):
    self.device = device

  def build_affine_matrix_2d(self, batch_size, params):
    """
    Return a affine transformation matrix
    size: size of input .size() method
    params: tuple of (s, o, a, z), where:
      s: sample scales  (bs, 2)
      o: sample offsets (bs, 2)
      a: sample angles  (bs, 1)
      z: sample shear   (bs, 2)
    """
    scale, offset, theta, shear = params
    ones = torch.ones(batch_size).float().to(self.device)
        
    # Scale
    Ms = torch.zeros([batch_size, 3, 3], 
                      device=self.device)
    Ms[:,0,0] = scale[:,0] 
    Ms[:,1,1] = scale[:,1] 
    Ms[:,2,2] = ones

    # Translation
    Mt = torch.zeros([batch_size, 3, 3], 
                      device=self.device)
    Mt[:,0,2] = offset[:,0]
    Mt[:,1,2] = offset[:,1]
    Mt[:,0,0] = ones
    Mt[:,1,1] = ones
    Mt[:,2,2] = ones

    # Rotation
    Mr = torch.zeros([batch_size, 3, 3], device=self.device)

    Mr[:,0,0] = torch.cos(theta[:,0])
    Mr[:,0,1] = -torch.sin(theta[:,0])
    Mr[:,1,0] = torch.sin(theta[:,0])
    Mr[:,1,1] = torch.cos(theta[:,0])
    Mr[:,2,2] = ones

    # Shear
    Mz = torch.zeros([batch_size, 3, 3], 
                      device=self.device)
    
    Mz[:,0,1] = shear[:,0]
    Mz[:,1,0] = shear[:,1]  
    Mz[:,0,0] = ones
    Mz[:,1,1] = ones
    Mz[:,2,2] = ones
    
    M = torch.bmm(Mz,torch.bmm(Ms,torch.bmm(Mt, Mr)))
    return M

  def deform_img(self, img, params):
    Ma = self.build_affine_matrix_2d(len(img), params)
    phi_inv = F.affine_grid(torch.inverse(Ma)[:,:2,:],
                              img.size(),
                              align_corners=False).cuda()   
    img_moved = F.grid_sample(img.cuda(),
                            grid=phi_inv,
                            padding_mode='border', 
                            align_corners=False) 

    return img_moved
    
  def deform_points(self, points, params):
    batch_size, num_points, dim = points.shape
    Ma = self.build_affine_matrix_2d(batch_size, params)
    points = torch.cat((points, torch.ones(batch_size, num_points, 1).to(self.device)), dim=-1)
    warp_points = torch.bmm(Ma[:,:2,:], points.permute(0,2,1)).permute(0,2,1)
    return warp_points

class AffineDeformation3d():
  def __init__(self, device='cuda:0'):
    self.device = device

  def build_affine_matrix_3d(self, batch_size, params):
    """
    Return a affine transformation matrix
    batch_size: size of batch
    params: tuple of torch.FloatTensor
      scales  (batch_size, 3)
      offsets (batch_size, 3)
      angles  (batch_size, 3)
      shear   (batch_size, 6)
    """
    scale, offset, theta, shear = params
        
    ones = torch.ones(batch_size).float().to(self.device)
    
    # Scaling
    Ms = torch.zeros([batch_size, 4, 4], 
                      device=self.device)
    Ms[:,0,0] = scale[:,0] 
    Ms[:,1,1] = scale[:,1] 
    Ms[:,2,2] = scale[:,2] 
    Ms[:,3,3] = ones

    # Translation
    Mt = torch.zeros([batch_size, 4, 4], 
                      device=self.device)
    Mt[:,0,3] = offset[:,0]
    Mt[:,1,3] = offset[:,1]
    Mt[:,2,3] = offset[:,2]   
    Mt[:,0,0] = ones
    Mt[:,1,1] = ones
    Mt[:,2,2] = ones
    Mt[:,3,3] = ones

    # Rotation
    dim1_matrix = torch.zeros([batch_size, 4, 4], device=self.device)
    dim2_matrix = torch.zeros([batch_size, 4, 4], device=self.device)
    dim3_matrix = torch.zeros([batch_size, 4, 4], device=self.device)

    dim1_matrix[:,0,0] = ones
    dim1_matrix[:,1,1] = torch.cos(theta[:,0])
    dim1_matrix[:,1,2] = -torch.sin(theta[:,0])
    dim1_matrix[:,2,1] = torch.sin(theta[:,0])
    dim1_matrix[:,2,2] = torch.cos(theta[:,0])
    dim1_matrix[:,3,3] = ones

    dim2_matrix[:,0,0] = torch.cos(theta[:,1])
    dim2_matrix[:,0,2] = torch.sin(theta[:,1])
    dim2_matrix[:,1,1] = ones
    dim2_matrix[:,2,0] = -torch.sin(theta[:,1])
    dim2_matrix[:,2,2] = torch.cos(theta[:,1])
    dim2_matrix[:,3,3] = ones

    dim3_matrix[:,0,0] = torch.cos(theta[:,2])
    dim3_matrix[:,0,1] = -torch.sin(theta[:,2])
    dim3_matrix[:,1,0] = torch.sin(theta[:,2])
    dim3_matrix[:,1,1] = torch.cos(theta[:,2])
    dim3_matrix[:,2,2] = ones
    dim3_matrix[:,3,3] = ones

    """Shear"""
    Mz = torch.zeros([batch_size, 4, 4], 
                      device=self.device)
    
    Mz[:,0,1] = shear[:,0]
    Mz[:,0,2] = shear[:,1]
    Mz[:,1,0] = shear[:,2]  
    Mz[:,1,2] = shear[:,3]
    Mz[:,2,0] = shear[:,4]
    Mz[:,2,1] = shear[:,5]
    Mz[:,0,0] = ones
    Mz[:,1,1] = ones
    Mz[:,2,2] = ones
    Mz[:,3,3] = ones
    
    Mr = torch.bmm(dim3_matrix, torch.bmm(dim2_matrix, dim1_matrix))
    M = torch.bmm(Mz,torch.bmm(Ms,torch.bmm(Mt, Mr)))
    return M
  
  def deform_img(self, img, params, interp_mode='bilinear'):
    Ma = self.build_affine_matrix_3d(len(img), params)
    phi_inv = F.affine_grid(torch.inverse(Ma)[:,:3,:],
                              img.size(),
                              align_corners=False).to(self.device)
    img_moved = F.grid_sample(img.to(self.device),
                            grid=phi_inv,
                            mode=interp_mode,
                            padding_mode='border', 
                            align_corners=False) 
    return img_moved
  
  def deform_points(self, points, params):
    batch_size, num_points, dim = points.shape
    Ma = self.build_affine_matrix_3d(batch_size, params)
    points = torch.cat((points, torch.ones(batch_size, num_points, 1).to(self.device)), dim=-1)
    warp_points = torch.bmm(Ma[:,:3,:], points.permute(0,2,1)).permute(0,2,1)
    return warp_points
  
  def __call__(self, img, **kwargs):
    params = kwargs['params']
    interp_mode = kwargs['interp_mode']
    return self.deform_img(img, params, interp_mode)

# Functions
def augment_moving(x, args, seg=None, max_random_params=(0.2, 0.2, 3.1416, 0.1), fixed_params=None):
    '''Augment moving image and corresponding segmentation.
    
    max_random_params: 4-tuple of floats, max value of each transformation for random augmentation
    fixed_params: Fixed parameters for transformation. Fixed augmentation if set.
    '''
    if fixed_params:
        s, o, a, z = fixed_params
        if args.dim == 2:
            scale = torch.tensor([e+1 for e in s]).unsqueeze(0).float()
            offset = torch.tensor(o).unsqueeze(0).float()
            theta = torch.tensor(a).unsqueeze(0).float()
            shear = torch.tensor(z).unsqueeze(0).float()
        else:
            scale = torch.tensor([e+1 for e in s]).unsqueeze(0).float()
            offset = torch.tensor(o).unsqueeze(0).float()
            theta = torch.tensor(a).unsqueeze(0).float()
            shear = torch.tensor(z).unsqueeze(0).float()
    else:
        s, o, a, z = max_random_params
        if args.dim == 2:
            scale = torch.FloatTensor(1, 2).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 2).uniform_(-o, o)
            theta = torch.FloatTensor(1, 1).uniform_(-a, a)
            shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        else:
            scale = torch.FloatTensor(1, 3).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 3).uniform_(-o, o)
            theta = torch.FloatTensor(1, 3).uniform_(-a, a)
            shear = torch.FloatTensor(1, 6).uniform_(-z, z)

    params = (scale, offset, theta, shear)

    if args.dim == 2:
        augmenter = AffineDeformation2d(device=args.device)
    else:
        augmenter = AffineDeformation3d(device=args.device)
    x = augmenter(x, params=params, interp_mode='bilinear')
    if seg is not None:
        seg = augmenter(seg, params=params, interp_mode='nearest')
        return x, seg
    return x

def augment_pair(x1, x2, args, params=(0.2, 0.2, 3.1416, 0.1), random=True):
    s, o, a, z = params
    if random:
        if args.dim == 2:
            scale = torch.FloatTensor(1, 2).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 2).uniform_(-o, o)
            theta = torch.FloatTensor(1, 1).uniform_(-a, a)
            shear = torch.FloatTensor(1, 2).uniform_(-z, z)
        else:
            scale = torch.FloatTensor(1, 3).uniform_(1-s, 1+s)
            offset = torch.FloatTensor(1, 3).uniform_(-o, o)
            theta = torch.FloatTensor(1, 3).uniform_(-a, a)
            shear = torch.FloatTensor(1, 6).uniform_(-z, z)
    else:
        if args.dim == 2:
            scale = torch.FloatTensor(1, 2).fill_(1-s, 1+s)
            offset = torch.FloatTensor(1, 2).fill_(-o, o)
            theta = torch.FloatTensor(1, 1).fill_(-a, a)
            shear = torch.FloatTensor(1, 2).fill_(-z, z)
        else:
            scale = torch.FloatTensor(1, 3).fill_(1+s)
            offset = torch.FloatTensor(1, 3).fill_(o)
            theta = torch.FloatTensor(1, 3).fill_(a)
            shear = torch.FloatTensor(1, 6).fill_(z)

    params = (scale, offset, theta, shear)

    if args.dim == 2:
        augmenter = AffineDeformation2d(device=args.device)
    else:
        augmenter = AffineDeformation3d(device=args.device)
    x1 = augmenter(x1, params=params, interp_mode='bilinear')
    x2 = augmenter(x2, params=params, interp_mode='bilinear')
    return x1, x2

def augment_moving_points(x_fixed, points, args, params=(0.2, 0.2, 3.1416, 0.1)):
    s, o, a, z = params
    s = np.clip(s*args.epoch / args.affine_slope, None, s)
    o = np.clip(o*args.epoch / args.affine_slope, None, o)
    a = np.clip(a*args.epoch / args.affine_slope, None, a)
    z = np.clip(z*args.epoch / args.affine_slope, None, z)
    if args.dim == 2:
        scale = torch.FloatTensor(1, 2).uniform_(1-s, 1+s)
        offset = torch.FloatTensor(1, 2).uniform_(-o, o)
        theta = torch.FloatTensor(1, 1).uniform_(-a, a)
        shear = torch.FloatTensor(1, 2).uniform_(-z, z)
    else:
        scale = torch.FloatTensor(1, 3).uniform_(1-s, 1+s)
        offset = torch.FloatTensor(1, 3).uniform_(-o, o)
        theta = torch.FloatTensor(1, 3).uniform_(-a, a)
        shear = torch.FloatTensor(1, 6).uniform_(-z, z)

    params = (scale, offset, theta, shear)

    if args.dim == 2:
        augmenter = AffineDeformation2d(device=args.device)
    else:
        augmenter = AffineDeformation3d(device=args.device)
    x_moving = augmenter.deform_img(x_fixed, params)
    points = augmenter.deform_points(points, params)
    return x_moving, points
