import os
import torch
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path

from keymorph import keypoint_aligners as rt
from keymorph.model import ConvNetFC, ConvNetCoM
from keymorph.step import step
import nibabel as nib

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
                        dest="num_keypoints",
                        default=64,
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
    # Data
    parser.add_argument('--moving', 
                        type=str,
                        required=True,
                        help='Moving image path')

    parser.add_argument('--fixed', 
                        type=str,
                        required=True,
                        help='Fixed image path')

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

def _validate_shape(img):
    assert len(img) >= 3 and len(img.shape) <= 5
    if len(img.shape) == 3:
        img = img.unsqueeze(0).unsqueeze(0)
    elif len(img.shape) == 4:
        img = img.unsqueeze(0)
    return img

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
    img_m = torch.tensor(nib.load(args.moving).get_fdata())
    img_f = torch.tensor(nib.load(args.fixed).get_fdata())
    img_m = _validate_shape(img_m)
    img_f = _validate_shape(img_f)

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
        kp_aligner = rt.ClosedFormAffine(args.dim)
    elif args.kp_align_method == 'tps':
        kp_aligner = rt.TPS(args.dim)
    else:
      raise NotImplementedError

    network.eval()

    metrics, imgs, points, grid = step(img_f, img_m, 
                                              network, 
                                              kp_aligner, 
                                              args, 
                                              is_train=False)
    img_f, img_m, img_a = imgs
    points_f, points_m, points_a = points

    for name, metric in metrics.items():
        print(f'[Eval Stat] {name}: {metric:.5f}')