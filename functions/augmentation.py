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