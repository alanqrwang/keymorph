import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from argparse import ArgumentParser
import torchio as tio

from keymorph.keypoint_aligners import ClosedFormAffine, TPS
from keymorph.model import ConvNetFC, ConvNetCoM
from keymorph.step import step

def parse_args():
    parser = ArgumentParser()

    # I/O
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which GPUs to use? Index from 0")

    parser.add_argument("--save_dir",
                        type=str,
                        dest="save_dir",
                        default="./training_output/",
                        help="Path to the folder where outputs are saved")

    parser.add_argument('--load_path', type=str, default=None,
              help='Load checkpoint at .h5 path')

    parser.add_argument("--save_preds",
                        action='store_true',
                        help='Perform evaluation')

    parser.add_argument("--visualize",
                        action='store_true',
                        help='Visualize images and points')

    # KeyMorph
    parser.add_argument("--num_keypoints",
                        type=int,
                        required=True,
                        help="Number of keypoints")

    parser.add_argument('--kp_extractor', 
                        type=str,
                        default='conv_com', 
                        choices=['conv_fc', 'conv_com'], 
                        help='Keypoint extractor module to use')

    parser.add_argument('--kp_align_method', 
                        type=str,
                        default='affine', 
                        choices=['affine', 'tps'], 
                        help='Keypoint alignment module to use')

    parser.add_argument('--tps_lmbda', 
                        default=None, 
                        help='TPS lambda value')

    parser.add_argument("--norm_type",
                        type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch', 'group'],
                        help="Normalization type")

    parser.add_argument("--loss_fn",
                        type=str,
                        default='mse',
                        choices=['mse', 'dice'],
                        help="Loss function")

    # Data
    parser.add_argument('--moving', 
                        type=str,
                        required=True,
                        help='Moving image path')

    parser.add_argument('--fixed', 
                        type=str,
                        required=True,
                        help='Fixed image path')
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size")

    # Miscellaneous
    parser.add_argument("--seed",
                        type=int,
                        dest="seed",
                        default=23,
                        help="Random seed use to sort the training data")

    parser.add_argument('--dim', 
                        type=int,
                        default=3)


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpus))
    else:
        args.device = torch.device('cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('Number of GPUs: {}'.format(torch.cuda.device_count()))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    transform = tio.Compose([
    #                 RandomBiasField(),
    #                 RandomNoise(),
                    tio.Lambda(lambda x: x.permute(0,1,3,2)),
                    tio.Resize(128),
    ])
    fixed = [tio.Subject(img=tio.ScalarImage(args.fixed))]
    moving = [tio.Subject(img=tio.ScalarImage(args.moving))]
    fixed_dataset = tio.SubjectsDataset(fixed, transform=transform)
    moving_dataset = tio.SubjectsDataset(moving, transform=transform)
    fixed_loader = DataLoader(
                  fixed_dataset,
                  batch_size=args.batch_size,
                  shuffle=False,
    )
    moving_loader = DataLoader(
                  moving_dataset,
                  batch_size=args.batch_size,
                  shuffle=False,
    )

    # CNN, i.e. keypoint extractor
    if args.kp_extractor == 'conv_fc':
      network = ConvNetFC(args.dim,
                   1, 
                   args.num_keypoints*args.dim,
                   norm_type=args.norm_type)    
      network = torch.nn.DataParallel(network)
    elif args.kp_extractor == 'conv_com':
      network = ConvNetCoM(args.dim,
                   1, 
                   args.num_keypoints,
                   norm_type=args.norm_type)    
    network = torch.nn.DataParallel(network)
    network.to(args.device)

    if args.load_path:
        state_dict = torch.load(args.load_path)['state_dict']
        network.load_state_dict(state_dict)

    # Keypoint alignment module
    if args.kp_align_method == 'affine':
        kp_aligner = ClosedFormAffine(args.dim)
    elif args.kp_align_method == 'tps':
        kp_aligner = TPS(args.dim)
    else:
      raise NotImplementedError

    network.eval()

    for fixed in fixed_loader:
        for moving in moving_loader:
            metrics, imgs, points, grid = step(fixed, moving, 
                                              network, 
                                              kp_aligner, 
                                              args, 
                                              is_train=False)
            img_f, img_m, img_a = imgs
            points_f, points_m, points_a = points

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(img_f[0,0,64].cpu().numpy(), cmap='gray')
            axes[1].imshow(img_m[0,0,64].cpu().numpy(), cmap='gray')
            axes[2].imshow(img_a[0,0,64].cpu().numpy(), cmap='gray')
            plt.show()

            for name, metric in metrics.items():
                print(f'[Eval Stat] {name}: {metric:.5f}')