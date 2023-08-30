'''Input Libraries'''
import os
import torch
import numpy as np
import torchio as tio
import torch.nn.functional as F

from tqdm import tqdm
from keymorph import model as m
from argparse import ArgumentParser
from torchio.transforms import Lambda
from keymorph import cm_plotter as cp
from keymorph import augmentation as aug
from keymorph import loader_maker as loader


def sample_valid_coordinate(x, reg):
    """
    x: input mri with shape [1,1,dim1,dim2,dim3]
    reg: how many points within the brain
    """
    eps = 1e-1
    mask = x > eps
    indeces = []
    for i in range(reg):
        hit = 0
        while hit == 0:
            sample = torch.zeros_like(x)
            dim1 = np.random.randint(0, x.size(2))
            dim2 = np.random.randint(0, x.size(3))
            dim3 = np.random.randint(0, x.size(4))
            sample[:, :, dim1, dim2, dim3] = 1
            hit = (sample * mask).sum()
            if hit == 1:
                indeces += [[dim1, dim2, dim3]]

    grid = F.affine_grid(torch.Tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]),
                         x.size(),
                         align_corners=True)
    grid = grid.permute(0, 4, 1, 2, 3)

    coordinates = []
    for idx in indeces:
        coordinates += [grid[:, :, idx[0], idx[1], idx[2]].view(1, 3, 1)]

    return torch.cat(coordinates, -1)


def parser():
    parser = ArgumentParser()

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which GPUs to use? Index from 0")

    parser.add_argument("--out_dim",
                        type=int,
                        dest="out_dim",
                        default=64,
                        help="How many center masses")

    parser.add_argument("--batch_size",
                        type=int,
                        dest="batch_size",
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

    parser.add_argument("--epochs",
                        type=int,
                        dest="epochs",
                        default=2000,
                        help="Training Epochs")

    parser.add_argument("--downsample",
                        type=int,
                        dest="downsample",
                        default=2,
                        help="how much to downsample")

    parser.add_argument("--seed",
                        type=int,
                        dest="seed",
                        default=23,
                        help="Random seed use to sort the training data")

    parser.add_argument("--c",
                        type=int,
                        dest="c",
                        default=300,
                        help="Constant to control how slow to increase augmentation")

    parser.add_argument("--norm_type",
                        type=int,
                        default=1,
                        choices=[0, 1, 2])

    args = parser.parse_args()
    return args


def run(loader,
        tc,
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
    u1.train() if mode == 'train' else u1.eval()

    for j in range(steps_per_epoch):

        """Target"""
        # Choose modality at random
        mod = np.random.randint(3)
        data = iter(loader[mod]).next()

        x = data['mri'][tio.DATA]
        x_mask = data['mask'][tio.DATA].float()
        x = x.repeat(args.batch_size, 1, 1, 1, 1)
        x_mask = x_mask.repeat(args.batch_size, 1, 1, 1, 1)

        n_batch, _, dim1, dim2, dim3 = x.size()
        y = tc

        """Augment"""
        # Augment
        Ma = aug.affine_matrix(x.size(),
                               s=np.clip(0.2 * epoch / (args.c), None, 0.2),
                               o=np.clip(0.2 * epoch / (args.c), None, 0.2),
                               a=np.clip(3.1416 * epoch / (args.c), None, 3.1416),
                               z=np.clip(0.1 * epoch / (args.c), None, 0.1),
                               cuda=False)

        _grid = F.affine_grid(torch.inverse(Ma)[:, :3, :],
                              x.size(),
                              align_corners=False)

        x = F.grid_sample(x,
                          grid=_grid,
                          mode='bilinear',
                          padding_mode='border',
                          align_corners=False).detach()

        x_mask = F.grid_sample(x_mask,
                               grid=_grid,
                               mode='nearest',
                               padding_mode='border',
                               align_corners=False).detach()

        if not args.downsample == 1:
            x = F.avg_pool3d(x, args.downsample, args.downsample)
            x_mask = F.avg_pool3d(x_mask, args.downsample, args.downsample)

        x = x.cuda().requires_grad_()

        y = torch.cat([y, torch.ones(x.size(0), 1, args.out_dim)], 1)
        y = torch.bmm(Ma[:, :3, :], y)
        y = y.cuda()

        """Predict"""
        optimizer.zero_grad()
        x = x_mask.cuda() * x
        y_pred = u1(x)

        loss = F.mse_loss(y_pred, y.detach())

        if mode == 'train':
            loss.backward()
            optimizer.step()

        """Save"""
        running_loss += [loss.item()]

        x = x.cpu()
        y_pred = y_pred.cpu()
        y = y.cpu()

    loss = np.mean(running_loss)
    stat = [loss]

    size = 256 // args.downsample
    y_cm = cp.get_cm_plot(y, size, size, size)
    y_cm = cp.blur_cm_plot(y_cm, 1)

    pred_cm = cp.get_cm_plot(y_pred, size, size, size)
    pred_cm = cp.blur_cm_plot(pred_cm, 1)

    return stat


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    args = parser()
    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    """Load Data"""
    directory = './data/centered_IXI/'
    transform = Lambda(lambda x: x.permute(0, 1, 3, 2))

    train_set, t1_loader = loader.create(directory,
                                         start_end=[0, 1],
                                         modality='T1',
                                         transform=transform,
                                         batch_size=1,
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=0,
                                         seed=args.seed)

    _, t2_loader = loader.create(directory,
                                 start_end=[0, 1],
                                 modality='T2',
                                 transform=transform,
                                 batch_size=1,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0,
                                 seed=args.seed)

    _, pd_loader = loader.create(directory,
                                 start_end=[0, 1],
                                 modality='PD',
                                 transform=transform,
                                 batch_size=1,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=0,
                                 seed=args.seed)

    """Create a fix subject and target"""
    x_tg = train_set[0]['mask'][tio.DATA]
    x_tg = x_tg.float().unsqueeze(0)
    x_tg = F.avg_pool3d(x_tg, args.downsample, args.downsample)

    tc = sample_valid_coordinate(x_tg, args.out_dim)

    x_tg = x_tg.repeat(args.batch_size, 1, 1, 1, 1)
    tc = tc.repeat(args.batch_size, 1, 1)

    """Model"""
    u1 = m.KeyMorph(1, args.out_dim, args.norm_type)
    u1 = torch.nn.DataParallel(u1)
    u1.cuda()

    params = list(u1.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

    train_loss = []
    arguments = ('[pretrain]keypoints' + str(args.out_dim)
                 + '_batch' + str(args.batch_size)
                 + '_c' + str(args.c)
                 + '_lr' + str(args.lr)
                 + '_downsample' + str(args.downsample)
                 + '_normtype' + str(args.norm_type))

    PATH = args.save_dir + arguments + '/'

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        raise Exception('Path exists')

    '''Train'''
    best = 1e10
    val_loss = []
    for epoch in tqdm(range(args.epochs)):
        epoch = epoch
        stat = run(loader=[t1_loader, t2_loader, pd_loader],
                   tc=tc,
                   model=u1,
                   optimizer=optimizer,
                   epoch=epoch,
                   mode='train',
                   args=args,
                   PATH=PATH,
                   steps_per_epoch=32)

        train_loss += [stat[0]]

        print('Epoch %d' % (epoch))
        print('[Train Stat] Loss: %.5f'
              % (train_loss[-1]))

        train_summary = np.vstack((train_loss))

        '''Save model'''
        state = {'epoch': epoch,
                 'args': args,
                 'u1': u1.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'train_summary': train_summary}

        if train_loss[-1] < best:
            best = train_loss[-1]
            torch.save(state, PATH + 'pretrained_model.pth.tar')
        if (epoch + 1) % 49 == 0:
            torch.save(state, PATH + 'epoch{}_model.pth.tar'.format(epoch))
        del state
