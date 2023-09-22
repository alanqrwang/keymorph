import random
import os
import torch
import numpy as np
import torchio as tio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

from keymorph.net import ConvNetFC, ConvNetCoM
from keymorph.cm_plotter import show_warped, show_warped_vol
from keymorph.data import ixi
from keymorph import utils
from keymorph.augmentation import AffineDeformation2d, AffineDeformation3d

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which GPUs to use? Index from 0")

    parser.add_argument("--num_keypoints",
                        type=int,
                        default=128,
                        help="Number of keypoints")

    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-4,
                        help="Learning rate, default: 1e-4")

    parser.add_argument("--save_dir",
                        type=str,
                        dest="save_dir",
                        default='./pretraining_output/',
                        help="Path to the folder where data is saved")

    parser.add_argument("--data_dir",
                        type=str,
                        default="./data/centered_IXI/",
                        help="Path to the training data")

    parser.add_argument("--epochs",
                        type=int,
                        default=2000,
                        help="Training Epochs")

    parser.add_argument("--seed",
                        type=int,
                        default=23,
                        help="Random seed use to sort the training data")

    parser.add_argument("--affine_slope",
                        type=int,
                        default=100,
                        help="Constant to control how slow to increase augmentation")

    parser.add_argument("--norm_type",
                        type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch', 'group'],
                        help="Normalization type")

    parser.add_argument('--dim', 
                        type=int,
                        default=3)

    parser.add_argument("--debug_mode",
                        action='store_true',
                        help='Debug mode')

    parser.add_argument("--dataset",
                        type=str,
                        default='ixi',
                        help="Dataset")

    parser.add_argument("--num_test_subjects",
                        type=int,
                        default=100,
                        help="Number of test subjects")

    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Num workers")

    parser.add_argument("--visualize",
                        action='store_true',
                        help='Visualize images and points')

    parser.add_argument('--kp_extractor', 
                        type=str,
                        default='conv_com', 
                        choices=['conv_fc', 'conv_com'], 
                        help='Keypoint extractor module to use')

    parser.add_argument('--steps_per_epoch', 
                        type=int,
                        default=32, 
                        help='Number of gradient steps per epoch')

    parser.add_argument('--log_interval', 
                        type=int,
                        default=25, 
                        help='Frequency of logs')



    args = parser.parse_args()
    return args

def augment_moving(augmenter, x_fixed, points, epoch, args, s=0.2, o=0.2, a=3.1416, z=0.1):
    s = np.clip(s*epoch / args.affine_slope, None, s)
    o = np.clip(o*epoch / args.affine_slope, None, o)
    a = np.clip(a*epoch / args.affine_slope, None, a)
    z = np.clip(z*epoch / args.affine_slope, None, z)
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

    x_moving = augmenter.deform_img(x_fixed, params)
    tgt_points = augmenter.deform_points(points, params)
    return x_moving, tgt_points

def run_train(loaders,
              augmenter,
              random_points,
              network,
              optimizer,
              epoch,
              args):
    network.train()

    res = []

    random_points = random_points.to(args.device)
    for _ in range(args.steps_per_epoch):
        # Choose modality at random
        subject = next(iter(random.choice(loaders)))

        x_fixed = subject['img'][tio.DATA].float().to(args.device)

        # Deform image and fixed points
        x_moving, tgt_points = augment_moving(augmenter, x_fixed, random_points, epoch, args)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            pred_points = network(x_moving)
            pred_points = pred_points.view(-1, args.num_keypoints, args.dim)

            loss = F.mse_loss(tgt_points, pred_points)
            loss.backward()
            optimizer.step()

        # Compute metrics
        metrics = {}
        metrics['loss'] = loss.cpu().detach().numpy()
        res.append(metrics)
        
        if args.visualize:
            if args.dim == 2:
                show_warped(
                      x_moving[0,0].cpu().detach().numpy(),
                      x_fixed[0,0].cpu().detach().numpy(),
                      x_fixed[0,0].cpu().detach().numpy(),
                      tgt_points[0].cpu().detach().numpy(),
                      random_points[0].cpu().detach().numpy(), 
                      pred_points[0].cpu().detach().numpy())
            else:
                show_warped_vol(
                      x_moving[0,0].cpu().detach().numpy(),
                      x_fixed[0,0].cpu().detach().numpy(),
                      x_fixed[0,0].cpu().detach().numpy(),
                      tgt_points[0].cpu().detach().numpy(),
                      random_points[0].cpu().detach().numpy(), 
                      pred_points[0].cpu().detach().numpy())

    return utils.aggregate_dicts(res)

def main():
    args = parse_args()

    # Path to save outputs
    arguments = ('[training]keypoints' + str(args.num_keypoints)
                 + '_batch' + str(args.batch_size)
                 + '_normType' + str(args.norm_type)
                 + '_lr' + str(args.lr))

    save_path = Path(args.save_dir) / arguments
    if not os.path.exists(save_path) and not args.debug_mode:
        os.makedirs(save_path)

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpus))
    else:
        args.device = torch.device('cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('Number of GPUs: {}'.format(torch.cuda.device_count()))

    # Data
    if args.dataset == 'ixi':
        modalities = ['T1', 'T2', 'PD']
    
        transform = tio.Compose([
        #                 RandomBiasField(),
        #                 RandomNoise(),
                        tio.Lambda(lambda x: x.permute(0,1,3,2)),
                        tio.Mask(masking_method='mask'),
                        tio.Resize(128),
                        tio.Lambda(ixi.one_hot, include=('seg')),
        ])

        train_datasets = {}
        test_datasets = {}
        for mod in modalities:
            train_subjects = ixi.read_subjects_from_disk(args.data_dir, (0, 427), mod)
            train_datasets[mod] = tio.data.SubjectsDataset(train_subjects, transform=transform)
            test_subjects = ixi.read_subjects_from_disk(args.data_dir, (428, 428+args.num_test_subjects), mod)
            test_datasets[mod] = tio.data.SubjectsDataset(test_subjects, transform=transform)
    else:
        raise NotImplementedError

    fixed_loaders = {k : DataLoader(v, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.num_workers) for k, v in train_datasets.items()}

    # Get a single subject and extract random keypoints
    x_tg = train_datasets[modalities[0]][0]['mask'][tio.DATA]
    x_tg = x_tg.float().unsqueeze(0)

    print('sampling random keypoints...')
    random_points = utils.sample_valid_coordinates(x_tg, args.num_keypoints, args.dim)
    random_points = random_points*2-1
    random_points = random_points.repeat(args.batch_size, 1, 1)

    if args.dim == 2:
        augmenter = AffineDeformation2d(args.device)
    else:
        augmenter = AffineDeformation3d(args.device)

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

    # Optimizer
    params = list(network.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

    train_loss = []

    # Train
    network.train()
    best = 1e10
    for epoch in range(args.epochs):
        epoch = epoch
        epoch_stats = run_train(list(fixed_loaders.values()),
                                augmenter,
                                random_points,
                                network,
                                optimizer,
                                epoch,
                                args)

        train_loss.append(epoch_stats['loss'])

        print(f'Epoch {epoch}/{args.epochs}')
        for name, metric in epoch_stats.items():
            print(f'[Train Stat] {name}: {metric:.5f}')

        # Save model
        state = {'epoch': epoch,
                 'args': args,
                 'state_dict': network.state_dict(),
                 'optimizer': optimizer.state_dict()}

        if train_loss[-1] < best:
            best = train_loss[-1]
            torch.save(state, os.path.join(save_path, 'pretrained_model.pth.tar'))
        if epoch % args.log_interval == 0 and not args.debug_mode:
            torch.save(state, os.path.join(save_path, 'pretrained_epoch{}_model.pth.tar'.format(epoch)))
        del state

if __name__ == "__main__":
    main()