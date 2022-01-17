import torch
import skimage
from skimage.filters import gaussian

def blur_cm_plot(Cm_plot, sigma):
    """
    Blur the keypoints/center-of-masses for better visualiztion
    
    Arguments
    ---------
    Cm_plot : tensor with the center-of-masses
    sigma   : how much to blur

    Return
    ------
        out : blurred points
    """    
    
    n_batch = Cm_plot.shape[0]    
    n_reg = Cm_plot.shape[1]
    out = []
    for n in range(n_batch):
        cm_plot = Cm_plot[n,:,:,:]
        blur_cm_plot = []
        for r in range(n_reg):
            _blur_cm_plot = gaussian(cm_plot[r,:,:,:], 
                                     sigma=sigma,
                                     mode='nearest')
            _blur_cm_plot = torch.from_numpy(_blur_cm_plot).float().unsqueeze(0)
            blur_cm_plot += [_blur_cm_plot]

        blur_cm_plot = torch.cat(blur_cm_plot,0)
        out += [blur_cm_plot.unsqueeze(0)]
    return torch.cat(out,0)

def get_cm_plot(Y_cm, dim0, dim1, dim2):
    """
    Convert the coordinate of the keypoint/center-of-mass to points in an tensor
    
    Arguments
    ---------
    Y_cm : keypoints coordinates/center-of-masses[n_bath, 3, n_reg]
    dim  : dim of the image

    Return
    ------
        out : tensor it assigns value of 1 where keypoints are located otherwise 0
    """

    n_batch = Y_cm.shape[0]
    
    out = []
    for n in range(n_batch):
        Y = Y_cm[n,:,:]
        n_reg = Y.shape[1]

        axis2 = torch.linspace(-1,1,dim2).float()
        axis1 = torch.linspace(-1,1,dim1).float()
        axis0 = torch.linspace(-1,1,dim0).float()

        index0 = []
        for i in range(n_reg):
            index0.append(torch.argmin((axis0-Y[2,i])**2).item())

        index1 = []
        for i in range(n_reg):
            index1.append(torch.argmin((axis1-Y[1,i])**2).item())    

        index2 = []
        for i in range(n_reg):
            index2.append(torch.argmin((axis2-Y[0,i])**2).item())    

        cm_plot = torch.zeros(n_reg,dim0,dim1,dim2)
        for i in range(n_reg):
            cm_plot[i,index0[i],index1[i],index2[i]] = 1
            
        out += [cm_plot.unsqueeze(0)]
        
    return torch.cat(out,0)