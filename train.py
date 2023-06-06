import os
import torch
import numpy as np
import torch.nn.functional as F
import random
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.stats import loguniform
from pathlib import Path

from functions.loader_maker import get_loaders, get_loader_same_sub
from functions import registration_tools as rt
from functions import loss_ops
from functions.model import ConvNetFC, ConvNetCoM
from functions import utils
from functions.cm_plotter import show_warped, show_warped_vol

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which GPUs to use? Index from 0")

    parser.add_argument("--num_keypoints",
                        type=int,
                        dest="num_keypoints",
                        default=64,
                        help="Number of keypoints")

    parser.add_argument("--batch_size",
                        type=int,
                        dest="batch_size",
                        default=1,
                        help="Batch size")

    parser.add_argument("--norm_type",
                        type=str,
                        default='instance',
                        choices=['none', 'instance', 'batch', 'group'],
                        help="Normalization type")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=3e-6,
                        help="Learning rate")

    parser.add_argument("--seed",
                        type=int,
                        dest="seed",
                        default=23,
                        help="Random seed use to sort the training data")

    parser.add_argument("--save_dir",
                        type=str,
                        dest="save_dir",
                        default="./training_output/",
                        help="Path to the folder where outputs are saved")

    parser.add_argument("--data_dir",
                        type=str,
                        dest="data_dir",
                        default="./data/centered_IXI/",
                        help="Path to the training data")

    parser.add_argument("--epochs",
                        type=int,
                        dest="epochs",
                        default=2000,
                        help="Training Epochs")

    parser.add_argument("--downsample",
                        type=int,
                        dest="downsample",
                        default=2,
                        help="How much to downsample using average pool")

    parser.add_argument('--modalities', 
                        type=str,
                        nargs='+', 
                        default=('T1', 'T2', 'PD'))

    parser.add_argument("--mix_modalities",
                        action='store_true',
                        help='Whether or not to mix modalities amongst image pairs')

    parser.add_argument('--transform', type=str,
              default='none')

    parser.add_argument('--loss_fn', type=str, default='mse')

    parser.add_argument('--kp_align_method', type=str,
              default='affine', choices=['affine', 'tps'], help='Keypoint alignment module to use')

    parser.add_argument('--arch', type=str,
              default='conv_com', choices=['conv_fc', 'conv_com'], help='Keypoint extractor module to use')

    parser.add_argument('--dim', 
                        type=int,
                        default=3)

    parser.add_argument('--tps_lmbda', type=float,
              default=None, help='TPS lambda value')


    parser.add_argument('--steps_per_epoch', type=int,
              default=32, help='Number of gradient steps per epoch')

    parser.add_argument('--log_interval', type=int,
              default=25, help='Frequency of logs')

    parser.add_argument("--eval",
                        action='store_true',
                        help='Perform evaluation')

    parser.add_argument("--pretrain",
                        action='store_true',
                        help='Self-supervised pretraining')

    parser.add_argument('--load_path', type=str, default=None,
              help='Load checkpoint at .h5 path')

    parser.add_argument("--save_preds",
                        action='store_true',
                        help='Perform evaluation')

    parser.add_argument("--visualize",
                        action='store_true',
                        help='Visualize images and points')

    parser.add_argument("--kpconsistency",
                        action='store_true',
                        help='Minimize keypoint consistency loss')

    args = parser.parse_args()
    return args

def _get_lmbda(num_samples, is_train=True):
    if not is_train and args.tps_lmbda in ['uniform', 'lognormal', 'loguniform']:
      choices = [0, 0.01, 0.1, 1.0, 10]
      lmbda = torch.tensor(np.random.choice(choices, size=num_samples)).to(args.device)

    if args.tps_lmbda is None:
      assert args.kp_align_method != 'tps', 'Need to implement this'
      lmbda = None
    elif args.tps_lmbda == 'uniform':
      lmbda = torch.rand(num_samples).to(args.device) * 10
    elif args.tps_lmbda == 'lognormal':
      lmbda = torch.tensor(np.random.lognormal(size=num_samples)).to(args.device)
    elif args.tps_lmbda == 'loguniform':
      a, b = 1e-6, 10
      lmbda = torch.tensor(loguniform.rvs(a, b, size=num_samples)).to(args.device)
    else:
      lmbda = torch.tensor(args.tps_lmbda).repeat(num_samples).to(args.device)
    return lmbda

def step(img_f, img_m, seg_f, seg_m, network, optimizer, kp_aligner, args, aug_params=None, is_train=True):
    '''Forward pass for one mini-batch step. 
    Variables with (_f, _m, _a) denotes (fixed, moving, aligned).
    
    Args:
        img_f, img_m: Fixed and moving intensity image (bs, 1, l, w, h)
        seg_f, seg_m: Fixed and moving one-hot segmentation map (bs, num_classes, l, w, h)
        network: Feature extractor network
        optimizer: Optimizer
        kp_aligner: Affine or TPS keypoint alignment module
        args: Other script parameters
    '''
    if is_train:
       assert network.training

    img_f = img_f.float().to(args.device)
    img_m = img_m.float().to(args.device)
    seg_f = seg_f.float().to(args.device)
    seg_m = seg_m.float().to(args.device)

    img_m, seg_m = utils.augment_moving(img_m, seg_m, args, fixed_params=aug_params)

    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        points_f = network(img_f)
        points_m = network(img_m)
        points_f = points_f.view(-1, args.num_keypoints, args.dim)
        points_m = points_m.view(-1, args.num_keypoints, args.dim)

        if args.num_keypoints > 256: # Take mini-batch of keypoints
          key_batch_idx = np.random.choice(args.num_keypoints, size=256, replace=False)
          points_f = points_f[:, key_batch_idx]
          points_m = points_m[:, key_batch_idx]
        
        # Align via keypoints
        lmbda = _get_lmbda(len(points_m), is_train=is_train)
        grid = kp_aligner.grid_from_points(points_m, points_f, img_f.shape, lmbda=lmbda)
        img_a, seg_a = utils.align_moving_img(grid, img_m, seg_m)
        points_a = kp_aligner.points_from_points(points_m, points_f, points_m, lmbda=lmbda)

        # Compute metrics (remove the ones you don't want to make code faster)
        mse = loss_ops.MSELoss()(img_f, img_a)
        soft_dice = loss_ops.DiceLoss()(seg_a, seg_f)
        hard_dice = loss_ops.DiceLoss(hard=True)(seg_a, seg_f,
                                                ign_first_ch=True)
        hausd = loss_ops.hausdorff_distance(seg_a, seg_f)
        grid = grid.permute(0, 4, 1, 2, 3)
        jdstd = loss_ops.jdstd(grid)
        jdlessthan0 = loss_ops.jdlessthan0(grid, as_percentage=True)

        if args.loss_fn == 'mse':
          loss = mse
        elif args.loss_fn == 'dice':
          loss = soft_dice

        if is_train:
            # Backward pass
            loss.backward()
            optimizer.step()

        metrics = {
           'loss': loss.cpu().detach().numpy(),
           'mse': mse.cpu().detach().numpy(),
           'softdice': 1-soft_dice.cpu().detach().numpy(),
           'harddice': 1-hard_dice[0].cpu().detach().numpy(),
        #    'harddiceroi': 1-hard_dice[1].cpu().detach().numpy(),
           'hausd': hausd,
           'jdstd': jdstd,
           'jdlessthan0': jdlessthan0,
        }

    if args.visualize:
        if args.dim == 2:
            show_warped(
                    img_m[0,0].cpu().detach().numpy(),
                    img_a[0,0].cpu().detach().numpy(),
                    img_f[0,0].cpu().detach().numpy(),
                    seg_m[0,0].cpu().detach().numpy(),
                    seg_a[0,0].cpu().detach().numpy(),
                    seg_f[0,0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(), 
                    points_f[0].cpu().detach().numpy(),
                    save_dir='./training_output/',
                    save_name='0.png')
        else:
            show_warped_vol(
                    img_m[0,0].cpu().detach().numpy(),
                    img_f[0,0].cpu().detach().numpy(),
                    img_a[0,0].cpu().detach().numpy(),
                    seg_m[0,0].cpu().detach().numpy(),
                    seg_f[0,0].cpu().detach().numpy(),
                    seg_a[0,0].cpu().detach().numpy(),
                    points_m[0].cpu().detach().numpy(), 
                    points_f[0].cpu().detach().numpy(),
                    points_a[0].cpu().detach().numpy(),
                    save_dir='./training_output',
                    save_name='0.png')

    return metrics, \
           (img_f, img_m, img_a), \
           (seg_f, seg_m, seg_a), \
           (points_f, points_m, points_a), \
           grid

def kpconsistency_step(sub1, sub2, network, optimizer, args):
    sub1 = sub1.float().to(args.device)
    sub2 = sub2.float().to(args.device)
    sub1, sub2 = utils.augment_pair(sub1, sub2, args)

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        points1 = network(sub1)
        points2 = network(sub2)

        loss = loss_ops.MSELoss()(points1, points2)
        loss.backward()
        optimizer.step()

    return loss.cpu().detach().numpy()

def pretrain_step(x_fixed, x_moving, seg_fixed, seg_moving, network, optimizer):
    '''Pretrain for one step.'''
    del x_moving, seg_fixed, seg_moving

    x_fixed = x_fixed.float().to(args.device)
    # Deform image and fixed points
    x_moving, tgt_points = utils.augment_moving(x_fixed, args.random_points)

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
      pred_points = network(x_moving)
      pred_points = pred_points.view(-1, args.num_keypoints, args.dim)

      loss = F.mse_loss(tgt_points, pred_points)
      loss.backward()
      optimizer.step()
    
    if args.visualize:
      show_warped_vol(
                x_moving[0,0].cpu().detach().numpy(),
                x_fixed[0,0].cpu().detach().numpy(),
                x_fixed[0,0].cpu().detach().numpy(),
                None,
                None,
                None,
                tgt_points[0].cpu().detach().numpy(),
                args.random_points[0].cpu().detach().numpy(), 
                pred_points[0].cpu().detach().numpy(),
                save_dir='./training_output',
                save_name='0.png')
    return loss.cpu().detach().numpy(), len(x_fixed)

def run_train(loader,
        same_subject_loader,
        network,
        optimizer,
        kp_aligner,
        args,
        steps_per_epoch=None):
    '''Train for one epoch.
    
    Args:
        loader: Dataloader for (fixed, moving) pairs 
        same_subject_loader: Dataloader for keypoint consistency loss
        network: keypoint extractor
        optimizer: Pytorch optimizer
        kp_aligner: keypoint aligner
        args: Other script arguments
        steps_per_epoch: int, number of gradient steps per epoch
    ''' 
    network.train()

    res = []
    for i, (x_fixed, x_moving, seg_fixed, seg_moving) in tqdm(enumerate(loader), 
        total=min(len(loader), steps_per_epoch)):

        metrics, _, _, _, _ = step(x_fixed, x_moving, 
                                   seg_fixed, seg_moving, 
                                   network, optimizer, 
                                   kp_aligner, 
                                   args)
        res.append(metrics)

        # Keypoint consistency loss
        if args.kpconsistency:
            sub1, sub2 = iter(same_subject_loader).next()
            kploss = kpconsistency_step(sub1, sub2, network, optimizer, args)
            metrics['kploss'] = kploss

        if steps_per_epoch and i == steps_per_epoch:
            break

    return utils.aggregate_dicts(res)

if __name__ == "__main__":
    args = parse_args()
    if args.loss_fn == 'mse':
        assert not args.mix_modalities, 'MSE loss can\'t mix modalities'

    # Select GPU
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpus))
    else:
        args.device = torch.device('cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('Number of GPUs {}'.format(torch.cuda.device_count()))

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    train_loader, _, test_loader = get_loaders(args.data_dir, 
                                     args.seed, 
                                     args.batch_size, 
                                     args.modalities, 
                                     args.downsample, 
                                     num_val_subjects=3, 
                                     num_test_subjects=3, 
                                     mix_modalities=args.mix_modalities,
                                     transform=args.transform)
    same_subject_loader = get_loader_same_sub(args.data_dir,
                                              args.seed, 
                                              args.batch_size, 
                                              args.modalities,
                                              args.downsample)

    # Model
    h_dims = [32, 64, 64, 128, 128, 256, 256, 512]
    if args.arch == 'conv_fc':
      network = ConvNetFC(args.dim,
                   1, 
                   h_dims, 
                   args.num_keypoints*args.dim,
                   norm_type=args.norm_type)    
      network = torch.nn.DataParallel(network)
    elif args.arch == 'conv_com':
      network = ConvNetCoM(args.dim,
                   1, 
                   h_dims, 
                   args.num_keypoints,
                   norm_type=args.norm_type)    
    network = torch.nn.DataParallel(network)
    network.to(args.device)

    if args.load_path:
        state_dict = torch.load(args.load_path)['state_dict']
        network.load_state_dict(state_dict)

    # Optimizer
    params = list(network.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

    # Path to save outputs
    arguments = ('[training]keypoints' + str(args.num_keypoints)
                 + '_batch' + str(args.batch_size)
                 + '_normType' + str(args.norm_type)
                 + '_downsample' + str(args.downsample)
                 + '_lr' + str(args.lr))

    save_path = Path(args.save_dir + arguments + '/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Keypoint alignment module
    if args.kp_align_method == 'affine':
        kp_aligner = rt.ClosedFormAffine(args.dim)
    elif args.kp_align_method == 'tps':
        kp_aligner = rt.TPS(args.dim)
    else:
      raise NotImplementedError

    if args.eval:
        # Evaluate on test set
        network.eval()

        list_of_test_mods = [
            'T1_T1',
            'T2_T2',
            'PD_PD',
            #
            'T1_T2',
            'T1_PD',
            'T2_PD',
        ] 
        list_of_test_augs = [
            'rot0',
            'rot45',
            'rot90',
            'rot135',
            'rot180',
        ]

        for aug in list_of_test_augs:
            for mod in list_of_test_mods:
                mod1, mod2, param = utils.parse_test_metric(mod, aug)
                for i, (img_f, seg_f) in tqdm(enumerate(test_loader[mod1]), 
                    total=len(test_loader[mod1])):
                    for j, (img_m, seg_m) in enumerate(test_loader[mod2]):
                        metrics, imgs, segs, points, grid = step(img_f, img_m, 
                                                                 seg_f, seg_m, 
                                                                 network, optimizer, 
                                                                 kp_aligner, 
                                                                 args, aug_params=param, is_train=False)
                        print(f'Running test: subject id {i}->{j}, mod {mod1}->{mod2}, aug {aug}')
                        img_f, img_m, img_a = imgs
                        seg_f, seg_m, seg_a = segs
                        points_f, points_m, points_a = points

                        if args.save_preds:
                            assert args.batch_size == 1 # TODO: fix this
                            img_f_path =    save_path / 'data' / f'img_f_{i}-{mod1}.npy'
                            seg_f_path =    save_path / 'data' / f'seg_f_{i}-{mod1}.npy'
                            points_f_path = save_path / 'data' / f'points_f_{i}-{mod1}.npy'
                            img_m_path =    save_path / 'data' / f'img_m_{j}-{mod2}-{aug}.npy'
                            seg_m_path =    save_path / 'data' / f'seg_m_{j}-{mod2}-{aug}.npy'
                            points_m_path = save_path / 'data' / f'points_m_{j}-{mod2}-{aug}.npy'
                            img_a_path =    save_path / 'data' / f'img_a_{i}-{mod1}_{j}-{mod2}-{aug}.npy'
                            seg_a_path =    save_path / 'data' / f'seg_a_{i}-{mod1}_{j}-{mod2}-{aug}.npy'
                            points_a_path = save_path / 'data' / f'points_a_{i}-{mod1}_{j}-{mod2}-{aug}.npy'
                            grid_path =     save_path / 'data' / f'grid_{i}-{mod1}_{j}-{mod2}-{aug}.npy'
                            print('Saving:\n{}\n{}\n{}\n{}\n'.format(img_f_path, img_m_path, img_a_path, grid_path))
                            np.save(img_f_path, img_f[0].cpu().detach().numpy())
                            np.save(img_m_path, img_m[0].cpu().detach().numpy())
                            np.save(img_a_path, img_a[0].cpu().detach().numpy())
                            np.save(seg_f_path, np.argmax(seg_f.cpu().detach().numpy(), axis=1))
                            np.save(seg_m_path, np.argmax(seg_m.cpu().detach().numpy(), axis=1))
                            np.save(seg_a_path, np.argmax(seg_a.cpu().detach().numpy(), axis=1))
                            np.save(points_f_path, points_f[0].cpu().detach().numpy())
                            np.save(points_m_path, points_m[0].cpu().detach().numpy())
                            np.save(points_a_path, points_a[0].cpu().detach().numpy())
                            np.save(grid_path, grid[0].cpu().detach().numpy())

                        for name, metric in metrics.items():
                            print(f'[Eval Stat] {name}: {metric:.5f}')
    
    else:
        # Train
        network.train()
        train_loss = []
        best = 1e10
        for epoch in range(1, args.epochs+1):
            epoch_stats = run_train(train_loader,
                            same_subject_loader,
                            network,
                            optimizer,
                            kp_aligner,
                            args,
                            steps_per_epoch=args.steps_per_epoch)

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
                torch.save(state, os.path.join(save_path, 'best_trained_model.pth.tar'))
            if epoch % args.log_interval == 0:
                torch.save(state, os.path.join(save_path, 'epoch{}_trained_model.pth.tar'.format(epoch)))