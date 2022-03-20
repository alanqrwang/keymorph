import torch


def center_of_mass(x, pytorch_grid=True):
    """
    Center of mass layer
    
    Arguments
    ---------
    x : network output
    pytorch_grid : use PyTorch convention for grid (-1,1) 

    Return
    ------
        C : center of masses for each chs
    """
    
    n_batch, chs, dim1, dim2, dim3 = x.shape
    eps = 1e-8
    if pytorch_grid:
        arange1 = torch.linspace(-1,1,dim1).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arange2 = torch.linspace(-1,1,dim2).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arange3 = torch.linspace(-1,1,dim3).float().view(1,1,-1).repeat(n_batch, chs, 1)
    else:
        arange1 = torch.arange(dim1).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arange2 = torch.arange(dim2).float().view(1,1,-1).repeat(n_batch, chs, 1)
        arange3 = torch.arange(dim3).float().view(1,1,-1).repeat(n_batch, chs, 1)
    
    if x.is_cuda:
        arange1, arange2, arange3 = arange1.cuda(), arange2.cuda(), arange3.cuda()
        
    m1 = x.sum((3,4)) #mass along the dimN, shape [n_batch, chs, dimN] 
    M1 = m1.sum(-1, True) + eps #total mass along dimN

    m2 = x.sum((2,4))
    M2 = m2.sum(-1, True) + eps

    m3 = x.sum((2,3)) 
    M3 = m3.sum(-1, True) + eps

    c1 = (arange1*m1).sum(-1,True)/M1 #center of mass along dimN, shape [n_batch, chs, 1]
    c2 = (arange2*m2).sum(-1,True)/M2
    c3 = (arange3*m3).sum(-1,True)/M3

    C = torch.cat([c3,c2,c1],-1) #center of mass, shape [n_batch, chs, 3]
    return C.transpose(-2,-1)


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