# KeypointMorph: Robust Multi-modal Affine Registration via Unsupervised Keypoint Detection

Implementation of KeypointMorph, an unsupervised end-to-end learning-based image registration framework that relies on automatically detecting corresponding keypoints. Our core insight is straightforward: matching keypoints between images can be used to obtain the optimal transformation via a differentiable closed-form expression. We use this observation to drive the unsupervised learning of anatomically-consistent keypoints from images. This not only leads to substantially more robust registration but also yields better interpretability, since the keypoints reveal which parts of the image are driving the final alignment. Moreover, KeypointMorph can be designed to be equivariant under image translations and/or symmetric with respect to the input image ordering. We demonstrate the proposed framework in solving 3D affine registration of multi-modal brain MRI scans. Remarkably, we show that this strategy leads to consistent keypoints, even across modalities.

## Requirements
We tested our code in Python 3.8 and the following packages:
- pytorch 1.10
- torchvision 1.11
- jupyterlab 3.2.1
- ipywidgets 7.6.5
- simpleitk 2.1.1
- scikit-image 0.18.3
- scikit-learn 1.0.1
- seaborn 0.11.2
- tqdm 4.62.3
- torchio 0.18.62
- numpy 1.21.3
- h5py 3.5.0
- antspyx 0.3.1

## TLDR
Keypoint registration using close-form solution (equation 2) in the paper can be done as follows:

        from functions import registration_tools as rt

        # Predict keypoints
        # model ouputs coordinate for each keypoint 
        # this is a tensor [n_batch, 3, n_keypoints] values ranging -1 to 1 (pytorch grid convention)
        
        moving_kp = model(x_moving)
        target_kp = model(x_target)

        # Close form
        affine_matrix = rt.close_form_affine(moving_kp, target_kp)
        inv_matrix = torch.zeros(x_moving.size(0),4,4)
        inv_matrix[:,:3,:4] = affine_matrix
        inv_matrix[:,3,3] = 1
        inv_matrix = torch.inverse(inv_matrix)[:,:3,:]
        grid = F.affine_grid(inv_matrix,
                             x.size(),
                             align_corners=False)
        
        # Align
        x_aligned = F.grid_sample(x,
                                  grid=grid,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)


## Step-by-Step Guide
