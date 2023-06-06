import torch
import torch.nn.functional as F

def close_form_affine(moving_kp, target_kp):
    """
    Obtain affine matrix to align moving keypoints to target keypoints.
    Affine matrix computed in a close form solution. 
    
    Arguments
    ---------
    moving_kp : keypoints from the moving image [n_batch, 3, n_keypoints]
    target_kp : keypoints from the fixed/target image [n_batch, 3, n_keypoints]

    Return
    ------
        out : affine matrix [n_batch, 3, 4]
    """
    Y_cm = moving_kp
    Y_tg = target_kp
    
    # Initialize 
    one = torch.ones(Y_cm.shape[0], 1, Y_cm.shape[2]).float() #Add a row of ones
    one = one.cuda() if Y_cm.is_cuda else one 
    _Y_cm = torch.cat([Y_cm, one],1)    
    
    out = torch.bmm(_Y_cm, torch.transpose(_Y_cm,-2,-1))
    out = torch.inverse(out)
    out = torch.bmm(torch.transpose(_Y_cm,-2,-1), out)
    out = torch.bmm(Y_tg, out)
    return out

class ClosedFormAffine:
  def __init__(self, dim):
    self.dim = dim

  def get_affine_matrix(self, x, y):
    """
    Solve the closed-form affine equation: A = y x^T (x x^T)^(-1).
    A is the solution to argmin_A ||Ax - y||_F

    Args:
      x, y: [n_batch, n_points, dim]
    Returns:
      A: [n_batch, 3, 4]
    """
    x = x.permute(0,2,1)
    y = y.permute(0,2,1)

    # Convert y to homogenous coordinates
    one = torch.ones(x.shape[0], 1, x.shape[2]).float().to(x.device) 
    x = torch.cat([x, one],1)    
    
    out = torch.bmm(x, torch.transpose(x,-2,-1))
    out = torch.inverse(out)
    out = torch.bmm(torch.transpose(x,-2,-1), out)
    out = torch.bmm(y, out)
    return out

  def grid_from_points(self, moving_points, fixed_points, grid_shape, **kwargs):
    del kwargs
    affine_matrix = self.get_affine_matrix(fixed_points, moving_points)
    grid = F.affine_grid(affine_matrix,
                          grid_shape,
                          align_corners=False)
    return grid
  
  def deform_points(self, points, matrix):
    square_mat = torch.zeros(len(points),self.dim+1,self.dim+1).to(points.device)
    square_mat[:,:self.dim,:self.dim+1] = matrix
    square_mat[:,-1,-1] = 1
    batch_size, num_points, _ = points.shape

    points = torch.cat((points, torch.ones(batch_size, num_points, 1).to(points.device)), dim=-1)
    warp_points = torch.bmm(square_mat[:,:3,:], points.permute(0,2,1)).permute(0,2,1)
    return warp_points

  def points_from_points(self, moving_points, fixed_points, points, **kwargs):
    affine_matrix = self.get_affine_matrix(moving_points, fixed_points)
    square_mat = torch.zeros(len(points),self.dim+1,self.dim+1).to(moving_points.device)
    square_mat[:,:self.dim,:self.dim+1] = affine_matrix
    square_mat[:,-1,-1] = 1
    batch_size, num_points, _ = points.shape

    points = torch.cat((points, torch.ones(batch_size, num_points, 1).to(moving_points.device)), dim=-1)
    warped_points = torch.bmm(square_mat[:,:3,:], points.permute(0,2,1)).permute(0,2,1)
    return warped_points


class TPS:       
  '''See https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/numpy.py'''
  def __init__(self, dim):
    self.dim = dim

  def fit(self, c, lmbda):        
    '''Assumes last dimension of c contains target points.
    
       Set up and solve linear system:
        [K   P] [w] = [v]
        [P^T 0] [a]   [0]
    Args:
      c: control points and target point (bs, T, d+1)
      lmbda: Lambda values per batch (bs)
    '''
    device = c.device
    bs, T = c.shape[0], c.shape[1]
    ctrl, tgt = c[:, :, :self.dim], c[:, :, -1]

    # Build K matrix
    U = TPS.u(TPS.d(ctrl, ctrl))
    I = torch.eye(T).repeat(bs, 1, 1).float().to(device)
    K = U + I*lmbda.view(bs, 1, 1)

    # Build P matrix
    P = torch.ones((bs, T, self.dim+1)).float()
    P[:, :, 1:] = ctrl

    # Build v vector
    v = torch.zeros(bs, T+self.dim+1).float()
    v[:, :T] = tgt

    A = torch.zeros((bs, T+self.dim+1, T+self.dim+1)).float()
    A[:, :T, :T] = K
    A[:, :T, -(self.dim+1):] = P
    A[:, -(self.dim+1):, :T] = P.transpose(1,2)

    theta = torch.linalg.solve(A, v) # p has structure w,a
    return theta
  
  @staticmethod
  def d(a, b):
    '''Compute pair-wise distances between points.
    
    Args:
      a: (bs, num_points, d)
      b: (bs, num_points, d)
    Returns:
      dist: (bs, num_points, num_points)
    '''
    return torch.sqrt(torch.square(a[:, :, None, :] - b[:, None, :, :]).sum(-1) + 1e-6)

  @staticmethod
  def u(r):
    '''Compute radial basis function.'''
    return r**2 * torch.log(r + 1e-6)
  
  def tps_theta_from_points(self, c_src, c_dst, lmbda):
    '''
    Args:
      c_src: (bs, T, dim)
      c_dst: (bs, T, dim)
      lmbda: (bs)
    '''
    device = c_src.device
    
    cx = torch.cat((c_src, c_dst[..., 0:1]), dim=-1)
    cy = torch.cat((c_src, c_dst[..., 1:2]), dim=-1)
    if self.dim == 3:
      cz = torch.cat((c_src, c_dst[..., 2:3]), dim=-1)

    theta_dx = self.fit(cx, lmbda).to(device)
    theta_dy = self.fit(cy, lmbda).to(device)
    if self.dim == 3:
      theta_dz = self.fit(cz, lmbda).to(device)

    if self.dim == 3:
      return torch.stack((theta_dx, theta_dy, theta_dz), -1)
    else:
      return torch.stack((theta_dx, theta_dy), -1)

  def tps(self, theta, ctrl, grid):
    '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
    The TPS surface is a minimum bend interpolation surface defined by a set of control points.
    The function value for a x,y location is given by
    
      TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
      
    This method computes the TPS value for multiple batches over multiple grid locations for 2 
    surfaces in one go.
    
    Params
    ------
    theta: Nx(T+3)xd tensor, or Nx(T+2)xd tensor
      Batch size N, T+3 model parameters for T control points in dx and dy.
    ctrl: NxTxd tensor
      T control points in normalized image coordinates [0..1]
    grid: NxHxWx(d+1) tensor
      Grid locations to evaluate with homogeneous 1 in first coordinate.
      
    Returns
    -------
    z: NxHxWxd tensor
      Function values at each grid location in dx and dy.
    '''
    
    if len(grid.shape) == 4:
      N, H, W, _ = grid.size()
      diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
    else:
      N, D, H, W, _ = grid.size()
      diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    T = ctrl.shape[1]
    
    pair_dist = torch.sqrt((diff**2).sum(-1))
    U = TPS.u(pair_dist)

    w, a = theta[:, :-(self.dim+1), :], theta[:, -(self.dim+1):, :]

    # U is NxHxWxT
    # b contains dot product of each kernel weight and U(r)
    b = torch.bmm(U.view(N, -1, T), w)
    if len(grid.shape) == 4:
      b = b.view(N,H,W,self.dim)
    else:
      b = b.view(N,D,H,W,self.dim)
    
    # b is NxHxWxd
    # z contains dot product of each affine term and polynomial terms.
    z = torch.bmm(grid.view(N,-1,self.dim+1), a)
    if len(grid.shape) == 4:
      z = z.view(N,H,W,self.dim) + b
    else:
      z = z.view(N,D,H,W,self.dim) + b
    return z

  def tps_grid(self, theta, ctrl, size):
    '''Compute a thin-plate-spline grid from parameters for sampling.
    
    Params
    ------
    theta: Nx(T+3)x2 tensor
      Batch size N, T+3 model parameters for T control points in dx and dy.
    ctrl: NxTx2 tensor, or Tx2 tensor
      T control points in normalized image coordinates [0..1]
    size: tuple
      Output grid size as NxCxHxW. C unused. This defines the output image
      size when sampling.
    
    Returns
    -------
    grid : NxHxWx2 tensor
      Grid suitable for sampling in pytorch containing source image
      locations for each output pixel.
    '''    
    device = theta.device
    if len(size) == 4:
      N, _, H, W = size
      grid_shape = (N, H, W, self.dim+1)
    else:
      N, _, D, H, W = size
      grid_shape = (N, D, H, W, self.dim+1)
    grid = self.uniform_grid(grid_shape).to(device)
    
    z = self.tps(theta, ctrl, grid)
    return z 

  def uniform_grid(self, shape):
    '''Uniform grid coordinates.
    
    Params
    ------
    shape : tuple
        NxHxWx3 defining the batch size, height and width dimension of the grid.
        3 is for the number of dimensions (2) plus 1 for the homogeneous coordinate.
    Returns
    -------
    grid: HxWx3 tensor
        Grid coordinates over [-1,1] normalized image range.
        Homogenous coordinate in first coordinate position.
        After that, the second coordinate varies first, then
        the third coordinate varies, then (optionally) the 
        fourth coordinate varies.
    '''

    if self.dim == 2:
      _, H, W, _ = shape
    else:
      _, D, H, W, _ = shape
    grid = torch.zeros(shape)

    grid[..., 0] = 1.
    grid[..., 1] = torch.linspace(-1, 1, W)
    grid[..., 2] = torch.linspace(-1, 1, H).unsqueeze(-1)   
    if grid.shape[-1] == 4:
      grid[..., 3] = torch.linspace(-1, 1, D).unsqueeze(-1).unsqueeze(-1)  
    return grid
  
  def grid_from_points(self, ctl_points, tgt_points, grid_shape, **kwargs):
    lmbda = kwargs['lmbda']

    theta = self.tps_theta_from_points(tgt_points, ctl_points, lmbda)
    grid = self.tps_grid(theta, tgt_points, grid_shape)
    return grid

  def deform_points(self, theta, ctrl, points):
    weights, affine = theta[:, :-(self.dim+1), :], theta[:, -(self.dim+1):, :]
    N, T, _ = ctrl.shape
    U = TPS.u(TPS.d(ctrl, points))

    P = torch.ones((N, points.shape[1], self.dim+1)).float().to(theta.device)
    P[:, :, 1:] = points[:, :, :self.dim]

    # U is NxHxWxT
    b = torch.bmm(U.transpose(1, 2), weights)
    z = torch.bmm(P.view(N,-1,self.dim+1), affine)
    return z + b
  
  def points_from_points(self, ctl_points, tgt_points, points, **kwargs):
    lmbda = kwargs['lmbda']
    theta = self.tps_theta_from_points(ctl_points, tgt_points, lmbda)
    return self.deform_points(theta, ctl_points, points)