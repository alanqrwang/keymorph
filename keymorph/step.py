import numpy as np

def step(img_f, img_m, 
         network, kp_aligner, 
         tps_lmbda,
         num_keypoints, 
         dim):
    '''Forward pass for one mini-batch step. 
    
    :param img_f, img_m: Fixed and moving images 
    :param network: Feature extractor network
    :param kp_aligner: Affine or TPS keypoint alignment module
    :param args: Other script parameters
    '''
    # Extract keypoints
    points_f, points_m = extract_keypoints_step(img_f, img_m, network)
    points_f = points_f.view(-1, num_keypoints, dim)
    points_m = points_m.view(-1, num_keypoints, dim)

    if num_keypoints > 256: # Take mini-batch of keypoints
        key_batch_idx = np.random.choice(num_keypoints, size=256, replace=False)
        points_f = points_f[:, key_batch_idx]
        points_m = points_m[:, key_batch_idx]
    
    # Align via keypoints
    grid = kp_aligner.grid_from_points(points_m, points_f, img_f.shape, lmbda=tps_lmbda)
    return grid, points_f, points_m

def extract_keypoints_step(img1, img2, network):
    return network(img1), network(img2)