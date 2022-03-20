'''Input Libraries'''
import os
import torch
import numpy as np
import torchio as tio
import torch.nn.functional as F

from tqdm import tqdm
from functions import model as m
from argparse import ArgumentParser
from torchio.transforms import Lambda
from functions import cm_plotter as cp
from functions import augmentation as aug
from functions import loader_maker as loader
from functions import registration_tools as rt


def train_parser():
    parser = ArgumentParser()

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which GPUs to use? Index from 0")

    parser.add_argument("--out_dim",
                        type=int,
                        dest="out_dim",
                        default=64,
                        help="Number of keypoints")

    parser.add_argument("--batch_size",
                        type=int,
                        dest="batch_size",
                        default=1,
                        help="Batch size")

    parser.add_argument("--norm_type",
                        type=int,
                        default=1,
                        choices=[0, 1, 2],
                        help="0:None, 1:Instance, 2:Batch")

    parser.add_argument("--lr",
                        type=float,
                        dest="lr",
                        default=1e-5,
                        help="Learning rate, default: 1e-5")

    parser.add_argument("--seed",
                        type=int,
                        dest="seed",
                        default=23,
                        help="Random seed use to sort the training data")

    parser.add_argument("--c",
                        type=int,
                        dest="c",
                        default=1,
                        help="Constant to control how slow to increase augmentation")

    parser.add_argument("--scratch",
                        action='store_true',
                        help='Train the model from scratch')

    parser.add_argument("--save_dir",
                        type=str,
                        dest="save_dir",
                        default="./training_output/",
                        help="Path to the folder where data is saved")

    parser.add_argument("--traing_dir",
                        type=str,
                        dest="traing_dir",
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

    parser.add_argument("--mod",
                        type=int,
                        dest="mod",
                        default=-1,
                        help="What modality to use for training. All:-1, T1:0, T2:1, PD:2")

    args = parser.parse_args()
    return args


def run(loader,
        model,
        optimizer,
        epoch,
        mode,
        args,
        PATH,
        steps_per_epoch):
    u1 = model
    running_loss = []

    """Choose samples"""
    u1.train()

    for j in range(steps_per_epoch):

        """Target"""
        # Choose modality at random
        mod = np.random.randint(3) if args.mod == -1 else args.mod
        fixed_data = iter(loader[mod]).next()

        x_tg = fixed_data['mri'][tio.DATA]
        xtg_mask = fixed_data['mask'][tio.DATA].float()
        if not args.downsample == 1:
            x_tg = F.avg_pool3d(x_tg, args.downsample, args.downsample)
            xtg_mask = F.avg_pool3d(xtg_mask, args.downsample, args.downsample)
        x_tg = x_tg.repeat(args.batch_size, 1, 1, 1, 1)
        xtg_mask = xtg_mask.repeat(args.batch_size, 1, 1, 1, 1)
        x_tg = x_tg.float().cuda()

        """Moving"""
        data = iter(loader[mod]).next()
        x = data['mri'][tio.DATA]
        x_mask = data['mask'][tio.DATA].float()
        x = x.repeat(args.batch_size, 1, 1, 1, 1)
        x_mask = x_mask.repeat(args.batch_size, 1, 1, 1, 1)

        """Augment"""
        # Augment
        Ma = aug.affine_matrix(x.size(),
                               s=np.clip(0.2 * epoch / (args.c), None, 0.2),
                               o=np.clip(0.2 * epoch / (args.c), None, 0.2),
                               a=np.clip(3.1416 * epoch / (args.c), None, 3.1416),
                               z=np.clip(0.1 * epoch / (args.c), None, 0.1),
                               cuda=False)

        grid = F.affine_grid(torch.inverse(Ma)[:, :3, :],
                             x.size(),
                             align_corners=False)

        x = F.grid_sample(x,
                          grid=grid,
                          mode='bilinear',
                          padding_mode='border',
                          align_corners=False).detach()

        x_mask = F.grid_sample(x_mask,
                               grid=grid,
                               mode='nearest',
                               padding_mode='border',
                               align_corners=False).detach()

        if not args.downsample == 1:
            x = F.avg_pool3d(x, args.downsample, args.downsample)
            x_mask = F.avg_pool3d(x_mask, args.downsample, args.downsample)

        x = x.cuda().requires_grad_()

        """Predict"""
        optimizer.zero_grad()

        x = x_mask.cuda() * x
        x_tg = xtg_mask.cuda() * x_tg

        moving_kp = u1(x)
        target_kp = u1(x_tg)

        """Close Form Affine Matrix"""
        affine_matrix = rt.close_form_affine(moving_kp, target_kp)
        inv_matrix = torch.zeros(x.size(0), 4, 4).cuda() if moving_kp.is_cuda else torch.zeros(x.size(0), 4, 4)
        inv_matrix[:, :3, :4] = affine_matrix
        inv_matrix[:, 3, 3] = 1
        inv_matrix = torch.inverse(inv_matrix)[:, :3, :]

        """Align Image"""
        grid = F.affine_grid(inv_matrix,
                             x.size(),
                             align_corners=False)

        x_aligned = F.grid_sample(x,
                                  grid=grid,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)

        loss = F.mse_loss(x_aligned, x_tg.detach())

        if mode == 'train':
            loss.backward()
            optimizer.step()

        """Save"""
        running_loss += [loss.item()]

    loss = np.mean(running_loss)
    stat = [loss]

    if PATH is not None:
        # Project the keypoints to 2D plane to visualize the result        
        size = 256 // args.downsample
        y_cm = torch.cat([moving_kp, torch.ones(moving_kp.size(0), 1, args.out_dim).cuda()], 1)
        y_cm = torch.bmm(affine_matrix[:, :3, :], y_cm)
        y_cm = cp.get_cm_plot(y_cm.cpu(), size, size, size)
        y_cm = cp.blur_cm_plot(y_cm, 1)

        tg_cm = cp.get_cm_plot(target_kp.cpu(), size, size, size)
        tg_cm = cp.blur_cm_plot(tg_cm, 1)

        vis.view_cm(x[:, :, :, :, [size // 2]],
                    x_aligned[:, :, :, :, [size // 2]],
                    x_tg[:, :, :, :, [size // 2]],
                    y_cm.sum(1, True).sum(-1, True),
                    tg_cm.sum(1, True).sum(-1, True),
                    epoch=epoch,
                    suffix='_' + mode,
                    image_idx=0,
                    PATH=PATH,
                    vmin=None,
                    vmax=None)
    return stat


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from functions import visualization as vis

    args = train_parser()

    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print('Number of GPUs {}'.format(torch.cuda.device_count()))

    """Load Data"""
    transform = Lambda(lambda x: x.permute(0, 1, 3, 2))
    directory = args.traing_dir
    N = len(os.listdir(directory + '/T1/'))  # use all subject in the folder.
    # Note that in our experiment we use start_end=[0,427] for training
    # start_end=[427,477] for validation
    # start_end=[477,577] for testing
    # these are the range of indeces use for training, validation and testing, respectively
    # feel free to change how to partition the dataset based on your experiment   

    _, t1_loader = loader.create(directory,
                                 start_end=[0, N],
                                 modality='T1',
                                 transform=transform,
                                 batch_size=1,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0,
                                 seed=args.seed)

    _, t2_loader = loader.create(directory,
                                 start_end=[0, N],
                                 modality='T2',
                                 transform=transform,
                                 batch_size=1,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0,
                                 seed=args.seed)

    _, pd_loader = loader.create(directory,
                                 start_end=[0, N],
                                 modality='PD',
                                 transform=transform,
                                 batch_size=1,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0,
                                 seed=args.seed)

    """Model"""
    u1 = m.KeyMorph(1, args.out_dim, args.norm_type)
    u1 = torch.nn.DataParallel(u1)
    u1.cuda()

    if not args.scratch:
        summary = torch.load('./weights/pretrained_model.pth.tar')['u1']
        u1.load_state_dict(summary)
        del summary

    params = list(u1.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

    old_epoch = 0
    train_loss = []
    arguments = ('[training]keypoints' + str(args.out_dim)
                 + '_batch' + str(args.batch_size)
                 + '_normType' + str(args.norm_type)
                 + '_downsample' + str(args.downsample)
                 + '_c' + str(args.c)
                 + '_lr' + str(args.lr))

    PATH = args.save_dir + arguments + '/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        raise Exception('Path exists')

    '''Train'''
    best = 1e10
    for epoch in tqdm(range(args.epochs)):
        mode = 'train'
        stat = run(loader=[t1_loader, t2_loader, pd_loader],
                   model=u1,
                   optimizer=optimizer,
                   epoch=epoch,
                   mode=mode,
                   args=args,
                   PATH=PATH,
                   steps_per_epoch=32)

        train_loss += [stat[0]]

        print('Epoch %d' % (epoch))
        print('[Train Stat] Loss: %.5f' % (train_loss[-1]))
        train_summary = np.vstack((train_loss))

        '''Save model'''
        state = {'epoch': epoch,
                 'args': args,
                 'u1': u1.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'train_summary': train_summary}

        if train_loss[-1] < best:
            best = train_loss[-1]
            torch.save(state, PATH + 'trained_model.pth.tar')
        if (epoch + 1) % 49 == 0:
            torch.save(state, PATH + 'epoch{}_model.pth.tar'.format(epoch))
        del state
