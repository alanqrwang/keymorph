import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from argparse import ArgumentParser
from pathlib import Path
import wandb
import torchio as tio
from scipy.stats import loguniform

from keymorph.keypoint_aligners import ClosedFormAffine, TPS
from keymorph import loss_ops
from keymorph.net import ConvNetFC, ConvNetCoM
from keymorph.model import KeyMorph
from keymorph import utils
from keymorph.utils import ParseKwargs, initialize_wandb, str_or_float, align_img
from keymorph.data import ixi
from keymorph.augmentation import augment_pair
from keymorph.augmentation import augment_moving
from keymorph.cm_plotter import show_warped, show_warped_vol

def parse_args():
    parser = ArgumentParser()

    # I/O
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which GPUs to use? Index from 0")

    parser.add_argument("--save_dir",
                        type=str,
                        default="./output/",
                        help="Path to the folder where outputs are saved")

    parser.add_argument("--data_dir",
                        type=str,
                        default="./data/centered_IXI/",
                        help="Path to the training data")

    parser.add_argument('--load_path', type=str, default=None,
              help='Load checkpoint at .h5 path')

    parser.add_argument("--save_preds",
                        action='store_true',
                        help='Perform evaluation')

    parser.add_argument("--visualize",
                        action='store_true',
                        help='Visualize images and points')


    parser.add_argument('--log_interval', 
                        type=int,
                        default=25, 
                        help='Frequency of logs')

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
                        type=str_or_float,
                        default=None, 
                        help='TPS lambda value')

    parser.add_argument("--kpconsistency_coeff",
                        type=float,
                        default=0, 
                        help='Minimize keypoint consistency loss')

    # Data
    parser.add_argument("--downsample",
                        type=int,
                        default=2,
                        help="How much to downsample using average pool")

    parser.add_argument("--mix_modalities",
                        action='store_true',
                        help='Whether or not to mix modalities amongst image pairs')

    parser.add_argument("--num_test_subjects",
                        type=int,
                        default=100,
                        help="How much to downsample using average pool")

    parser.add_argument("--num_workers",
                        type=int,
                        default=1,
                        help="Num workers")

    # ML
    parser.add_argument("--dataset",
                        type=str,
                        default='ixi',
                        help="Dataset")

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
                        default=3e-6,
                        help="Learning rate")

    parser.add_argument('--transform', 
                        type=str,
                        default='none')

    parser.add_argument('--loss_fn', 
                        type=str, 
                        default='mse')

    parser.add_argument("--epochs",
                        type=int,
                        default=2000,
                        help="Training Epochs")

    parser.add_argument('--steps_per_epoch', 
                        type=int,
                        default=32, 
                        help='Number of gradient steps per epoch')

    parser.add_argument("--eval",
                        action='store_true',
                        help='Perform evaluation')

    parser.add_argument("--debug_mode",
                        action='store_true',
                        help='Debug mode')

    # Miscellaneous
    parser.add_argument("--seed",
                        type=int,
                        default=23,
                        help="Random seed use to sort the training data")

    parser.add_argument('--dim', 
                        type=int,
                        default=3)

    # Weights & Biases
    parser.add_argument("--use_wandb",
                        action='store_true',
                        help='Use Wandb')
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    args = parser.parse_args()
    return args

def _get_tps_lmbda(num_samples, args, is_train=True):
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

def run_train(fixed_loaders,
              moving_loaders,
              registration_model,
              optimizer,
              kp_aligner,
              args):
    '''Train for one epoch.
    
    Args:
        fixed_loaders: list of Dataloaders for fixed images
        moving_loaders: list of Dataloaders for moving images
        network: keypoint extractor
        optimizer: Pytorch optimizer
        kp_aligner: keypoint aligner
        args: Other script arguments
    ''' 
    registration_model.train()

    res = []

    fixed_iters = [iter(loader) for loader in fixed_loaders]
    moving_iters = [iter(loader) for loader in moving_loaders]
    for _ in range(args.steps_per_epoch):
        if args.mix_modalities:
            fixed_iter = random.choice(fixed_iters)
            moving_iter = random.choice(moving_iters)
        else:
            mod_idx = np.random.randint(0, len(fixed_iters))
            fixed_iter = fixed_iters[mod_idx]
            moving_iter = moving_iters[mod_idx]
        
        fixed, moving = next(fixed_iter), next(moving_iter)

        # Get images and segmentations from TorchIO subject
        img_f, img_m = fixed['img'][tio.DATA], moving['img'][tio.DATA]
        if 'seg' in fixed and 'seg' in moving:
            seg_available = True
            seg_f, seg_m = fixed['seg'][tio.DATA], moving['seg'][tio.DATA]
        else:
            seg_available = False

        # Move to device
        img_f = img_f.float().to(args.device)
        img_m = img_m.float().to(args.device)
        if seg_available:
            seg_f = seg_f.float().to(args.device)
            seg_m = seg_m.float().to(args.device)

        # Explicitly augment moving image
        if seg_available:
            img_m, seg_m = augment_moving(img_m, args, seg=seg_m, fixed_params=None)
        else:
            img_m = augment_moving(img_m, args, fixed_params=None)

        lmbda = _get_tps_lmbda(len(img_f), args, is_train=True)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            grid, points_f, points_m = registration_model(img_f, img_m, 
                                                          lmbda,
                                                          )
        img_a = align_img(grid, img_m)
        if seg_available:
            seg_a = align_img(grid, seg_m)
        points_a = kp_aligner.points_from_points(points_m, points_f, points_m, lmbda=lmbda)

        # Compute metrics
        metrics = {}
        metrics['mse'] = loss_ops.MSELoss()(img_f, img_a)
        if seg_available:
            metrics['softdiceloss'] = loss_ops.DiceLoss()(seg_a, seg_f)
            metrics['softdice'] = 1-metrics['softdiceloss']
            metrics['harddice'] = 1-loss_ops.DiceLoss(hard=True)(seg_a, seg_f,
                                                    ign_first_ch=True)[0]
            if args.dim == 3: # TODO: Implement 2D metrics
                metrics['hausd'] = loss_ops.hausdorff_distance(seg_a, seg_f)
                grid = grid.permute(0, 4, 1, 2, 3)
                metrics['jdstd'] = loss_ops.jdstd(grid)
                metrics['jdlessthan0'] = loss_ops.jdlessthan0(grid, as_percentage=True)

        # Compute loss
        if args.loss_fn == 'mse':
          loss = metrics['mse']
        elif args.loss_fn == 'dice':
          loss = metrics['softdiceloss']
        metrics['loss'] = loss

        # Perform backward pass
        loss.backward()
        optimizer.step()

        # Keypoint consistency loss
        if args.kpconsistency_coeff > 0:
            mods = np.random.choice(len(moving_loaders), size=2, replace=False)
            rand_subject = np.random.randint(0, len(moving_iter))
            sub1 = moving_loaders[mods[0]].dataset[rand_subject]
            sub2 = moving_loaders[mods[1]].dataset[rand_subject]

            sub1 = sub1['img'][tio.DATA].float().to(args.device).unsqueeze(0)
            sub2 = sub2['img'][tio.DATA].float().to(args.device).unsqueeze(0)
            sub1, sub2 = augment_pair(sub1, sub2, args)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                points1, points2 = registration_model.extract_keypoints_step(sub1, sub2)

            kploss = args.kpconsistency_coeff * loss_ops.MSELoss()(points1, points2)
            kploss.backward()
            optimizer.step()
            metrics['kploss'] = kploss

        # Convert metrics to numpy
        metrics = {k: torch.as_tensor(v).detach().cpu().numpy().item() 
                      for k, v in metrics.items()}
        res.append(metrics)

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
                        )
            else:
                show_warped_vol(
                        img_m[0,0].cpu().detach().numpy(),
                        img_f[0,0].cpu().detach().numpy(),
                        img_a[0,0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(), 
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                        )
                show_warped_vol(
                        seg_m.argmax(1)[0].cpu().detach().numpy(),
                        seg_f.argmax(1)[0].cpu().detach().numpy(),
                        seg_a.argmax(1)[0].cpu().detach().numpy(),
                        points_m[0].cpu().detach().numpy(), 
                        points_f[0].cpu().detach().numpy(),
                        points_a[0].cpu().detach().numpy(),
                        )

    return utils.aggregate_dicts(res)

def print_dataset_stats(datasets, prefix=''):
    print(f'{prefix} dataset has {len(datasets)} modalities.')
    for mod_name, mod_dataset in datasets.items():
        print('-> Modality {} has {} subjects ({} images, {} masks and {} segmentations)'.format(
            mod_name,
            len(mod_dataset),
            sum('img' in s for s in mod_dataset.dry_iter()),
            sum('mask' in s for s in mod_dataset.dry_iter()),
            sum('seg' in s for s in mod_dataset.dry_iter()),
        ))

def main():
    args = parse_args()
    if args.loss_fn == 'mse':
        assert not args.mix_modalities, 'MSE loss can\'t mix modalities'
    if args.debug_mode:
        args.steps_per_epoch = 3

    # Path to save outputs
    arguments = ('[training]keypoints' + str(args.num_keypoints)
                 + '_batch' + str(args.batch_size)
                 + '_normType' + str(args.norm_type)
                 + '_downsample' + str(args.downsample)
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

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

        fixed_datasets = {}
        moving_datasets = {}
        test_datasets = {}
        for mod in modalities:
            train_subjects = ixi.read_subjects_from_disk(args.data_dir, (0, 427), mod)
            fixed_datasets[mod] = tio.data.SubjectsDataset(train_subjects, transform=transform)
            train_subjects = ixi.read_subjects_from_disk(args.data_dir, (0, 427), mod)
            moving_datasets[mod] = tio.data.SubjectsDataset(train_subjects, transform=transform)
            test_subjects = ixi.read_subjects_from_disk(args.data_dir, (428, 428+args.num_test_subjects), mod)
            test_datasets[mod] = tio.data.SubjectsDataset(test_subjects, transform=transform)
    else:
        raise NotImplementedError

    print_dataset_stats(fixed_datasets, 'Fixed train')
    print_dataset_stats(moving_datasets, 'Moving train')
    print_dataset_stats(test_datasets, 'Test')

    fixed_loaders = {k : DataLoader(v, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.num_workers) for k, v in fixed_datasets.items()}
    moving_loaders = {k : DataLoader(v, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=args.num_workers) for k, v in moving_datasets.items()}

    test_loaders = {k : DataLoader(v, 
                                   batch_size=args.batch_size, 
                                   shuffle=False) for k, v in test_datasets.items()}

    if args.kpconsistency_coeff > 0:
        assert len(moving_datasets) > 1, \
            'Need more than one modality to compute keypoint consistency loss'
        assert all([len(fd) == len(md) for fd, md in zip(fixed_datasets, moving_datasets)]), \
            'Must have same number of subjects for fixed and moving datasets'

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
    
    # Keypoint model
    registration_model = KeyMorph(network, kp_aligner, args.num_keypoints, args.dim)

    # Optimizer
    params = list(network.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

    if args.eval:
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
                for i, fixed in enumerate(test_loaders[mod1]):
                    for j, moving in enumerate(test_loaders[mod2]):
                        print(f'Running test: subject id {i}->{j}, mod {mod1}->{mod2}, aug {aug}')
                        img_f, img_m = fixed['img'][tio.DATA], moving['img'][tio.DATA]
                        if 'seg' in fixed and 'seg' in moving:
                            seg_available = True
                            seg_f, seg_m = fixed['seg'][tio.DATA], moving['seg'][tio.DATA]
                        else:
                            seg_available = False

                        # Move to device
                        img_f = img_f.float().to(args.device)
                        img_m = img_m.float().to(args.device)
                        if seg_available:
                            seg_f = seg_f.float().to(args.device)
                            seg_m = seg_m.float().to(args.device)

                        # Explicitly augment moving image
                        if seg_available:
                            img_m, seg_m = augment_moving(img_m, args, seg=seg_m, fixed_params=param)
                        else:
                            img_m = augment_moving(img_m, args, fixed_params=param)
                        lmbda = _get_tps_lmbda(len(img_f), args, is_train=False)

                        with torch.set_grad_enabled(False):
                            grid, points_f, points_m = registration_model(img_f, img_m, 
                                                                          lmbda,
                                                                          )
                        img_a = align_img(grid, img_m)
                        if seg_available:
                            seg_a = align_img(grid, seg_m)
                        points_a = kp_aligner.points_from_points(points_m, points_f, points_m, lmbda=lmbda)

                        # Compute metrics
                        metrics = {}
                        metrics['mse'] = loss_ops.MSELoss()(img_f, img_a)
                        if seg_available:
                            metrics['softdiceloss'] = loss_ops.DiceLoss()(seg_a, seg_f)
                            metrics['softdice'] = 1-metrics['softdiceloss']
                            metrics['harddice'] = 1-loss_ops.DiceLoss(hard=True)(seg_a, seg_f,
                                                                    ign_first_ch=True)[0]
                            if args.dim == 3: # TODO: Implement 2D metrics
                                metrics['hausd'] = loss_ops.hausdorff_distance(seg_a, seg_f)
                                grid = grid.permute(0, 4, 1, 2, 3)
                                metrics['jdstd'] = loss_ops.jdstd(grid)
                                metrics['jdlessthan0'] = loss_ops.jdlessthan0(grid, as_percentage=True)

                        if args.save_preds and not args.debug_mode:
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
        network.train()
        train_loss = []
        best = 1e10
        
        if args.use_wandb and not args.debug_mode:
            initialize_wandb(args)

        for epoch in range(1, args.epochs+1):
            epoch_stats = run_train(list(fixed_loaders.values()),
                                    list(moving_loaders.values()),
                                    registration_model,
                                    optimizer,
                                    kp_aligner,
                                    args)
            train_loss.append(epoch_stats['loss'])

            print(f'Epoch {epoch}/{args.epochs}')
            for name, metric in epoch_stats.items():
                print(f'[Train Stat] {name}: {metric:.5f}')

            if args.use_wandb and not args.debug_mode:
                wandb.log(epoch_stats)

            # Save model
            state = {'epoch': epoch,
                    'args': args,
                    'state_dict': network.state_dict(),
                    'optimizer': optimizer.state_dict()}

            if train_loss[-1] < best and not args.debug_mode:
                best = train_loss[-1]
                torch.save(state, os.path.join(save_path, 'best_trained_model.pth.tar'))
            if epoch % args.log_interval == 0 and not args.debug_mode:
                torch.save(state, os.path.join(save_path, 'epoch{}_trained_model.pth.tar'.format(epoch)))

if __name__ == "__main__":
    main()