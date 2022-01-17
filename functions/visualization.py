import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def view_cm(x_moved, 
            x_aligned,
            x, 
            cm_pred,
            cm_target,
            epoch, 
            suffix, 
            image_idx=0,
            PATH=None, 
            show_image=False, 
            vmin=0,
            vmax=1,
            titles=['Moved', 'Aligned', 'Tatget', 'CMs Pred', 'CMs Target']):
    
    """
    Plot images and keypoints
    
    Arguments
    ---------
    x_moved     : moving image
    x_aligned   : aligned image
    x           : target/fixed image
    cm_pred     : keypoints for the moving image
    cm_target   : keypoints for the target/fixed iamge
    epoch       : current epoch
    suffix      : string for naming the output images
    image_idx   : which image to plot out of the batch
    PATH        : where to save the image
    show_image  : display image
    vmin        : min value of the brain images
    vmax        : max value of the brain images
    titles      : title/name of each plot
    """    
    
    # Cross-section
    s = x.shape[-1]//2
    _x = x
    x = x[image_idx,0,:,:,s].data.cpu().numpy()
    x_moved = x_moved[image_idx,0,:,:,s].data.cpu().numpy()
    x_aligned = x_aligned[image_idx,0,:,:,s].data.cpu().numpy()
    cm_pred = cm_pred[image_idx,0,:,:,s].data.cpu().numpy()
    cm_target = cm_target[image_idx,0,:,:,s].data.cpu().numpy()
    
    fig, ax = plt.subplots(nrows=1, ncols=5)

    ax[0].set_title(titles[0])
    image = np.flipud(x_moved)
    ax[0].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].axis('off')
    
    ax[1].set_title(titles[1])
    image = np.flipud(x_aligned)
    ax[1].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].axis('off')
    
    ax[2].set_title(titles[2])
    image = np.flipud(x)
    ax[2].imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].axis('off')    

    ax[3].set_title(titles[3])
    image = np.flipud(cm_pred)
    ax[3].imshow(image, cmap='Blues')
    ax[3].axis('off')    
    
    ax[4].set_title(titles[4])
    image = np.flipud(cm_target)
    ax[4].imshow(image, cmap='Blues')
    ax[4].axis('off')    
    
    if PATH is not None:
        fig.set_size_inches(30, 12)
        fig.savefig(PATH+str(epoch)+suffix)
    elif show_image:
        plt.show()
        
    plt.close('all')