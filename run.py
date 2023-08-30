import os
import torch
import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
import wandb

from keymorph.data.ixi import get_loaders, get_loader_same_sub
from keymorph import keypoint_aligners as rt
from keymorph import loss_ops
from keymorph.model import ConvNetFC, ConvNetCoM
from keymorph import utils
from keymorph.step import step
from keymorph.utils import ParseKwargs, initialize_wandb
from keymorph.augmentation import augment_pair

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
                        default="./output/",
                        help="Path to the folder where outputs are saved")

    parser.add_argument("--data_dir",
                        type=str,
                        dest="data_dir",
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

    parser.add_argument("--kpconsistency_coeff",
                        type=float,
                        default=0, 
                        help='Minimize keypoint consistency loss')

    # Data
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

    # ML
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

    parser.add_argument('--transform', 
                        type=str,
                        default='none')

    parser.add_argument('--loss_fn', 
                        type=str, 
                        default='mse')

    parser.add_argument("--epochs",
                        type=int,
                        dest="epochs",
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
                        dest="seed",
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

def kpconsistency_step(sub1, sub2, network, optimizer, args):
    sub1 = sub1.float().to(args.device)
    sub2 = sub2.float().to(args.device)
    sub1, sub2 = augment_pair(sub1, sub2, args)

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        points1 = network(sub1)
        points2 = network(sub2)

        loss = args.kpconsistency_coeff * loss_ops.MSELoss()(points1, points2)
        loss.backward()
        optimizer.step()

    return loss.cpu().detach().numpy()

def run_train(loader,
              network,
              optimizer,
              kp_aligner,
              args,
              same_subject_loader=None,
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

    if args.kpconsistency_coeff > 0:
        assert same_subject_loader is not None, 'same_subject_loader must be provided if kpconsistency_coeff > 0'
        same_subject_iter = iter(same_subject_loader)
    res = []
    for i, (x_fixed, x_moving, seg_fixed, seg_moving) in tqdm(enumerate(loader), 
        total=min(len(loader), steps_per_epoch)):

        metrics, _, _, _, _ = step(x_fixed, x_moving, 
                                   network,  
                                   kp_aligner, 
                                   args,
                                   seg_f=seg_fixed, seg_m=seg_moving, 
                                   optimizer=optimizer,
                                   )
        res.append(metrics)

        # Keypoint consistency loss
        if args.kpconsistency_coeff > 0:
            sub1, sub2 = next(same_subject_iter)
            kploss = kpconsistency_step(sub1, sub2, network, optimizer, args)
            metrics['kploss'] = kploss

        if steps_per_epoch and i == steps_per_epoch:
            break

    return utils.aggregate_dicts(res)

if __name__ == "__main__":
    args = parse_args()
    if args.loss_fn == 'mse':
        assert not args.mix_modalities, 'MSE loss can\'t mix modalities'

    # Path to save outputs
    arguments = ('[training]keypoints' + str(args.num_keypoints)
                 + '_batch' + str(args.batch_size)
                 + '_normType' + str(args.norm_type)
                 + '_downsample' + str(args.downsample)
                 + '_lr' + str(args.lr))

    save_path = Path(args.save_dir) / arguments
    if not os.path.exists(save_path):
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
    train_loader, _, test_loader = get_loaders(args.data_dir, 
                                     args.seed, 
                                     args.batch_size, 
                                     args.modalities, 
                                     args.downsample, 
                                     num_val_subjects=3, 
                                     num_test_subjects=3, 
                                     mix_modalities=args.mix_modalities,
                                     transform=args.transform)
    if args.kpconsistency_coeff > 0:
        same_subject_loader = get_loader_same_sub(args.data_dir,
                                                  args.seed, 
                                                  args.batch_size, 
                                                  args.modalities,
                                                  args.downsample)
    else:
        same_subject_loader = None                                            

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

    # Optimizer
    params = list(network.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

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
                                                                 network, optimizer, 
                                                                 kp_aligner, 
                                                                 args, aug_params=param, 
                                                                 is_train=False,
                                                                 seg_f=seg_f, seg_m=seg_m, 
                                                                 )
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
        
        if args.use_wandb:
            initialize_wandb(args)

        for epoch in range(1, args.epochs+1):
            epoch_stats = run_train(train_loader,
                            network,
                            optimizer,
                            kp_aligner,
                            args,
                            same_subject_loader=same_subject_loader,
                            steps_per_epoch=args.steps_per_epoch)

            train_loss.append(epoch_stats['loss'])

            print(f'Epoch {epoch}/{args.epochs}')
            for name, metric in epoch_stats.items():
                print(f'[Train Stat] {name}: {metric:.5f}')

            if args.use_wandb:
                wandb.log(epoch_stats)

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