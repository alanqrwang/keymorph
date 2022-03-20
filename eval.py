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


def eval_parser():
    parser = ArgumentParser()

    parser.add_argument("input_mod",
                        type=int,
                        choices=[0, 1, 2],
                        help="Input Modality. T1:0, T2:1, PD:2")

    parser.add_argument("target_mod",
                        type=int,
                        choices=[0, 1, 2],
                        help="Target Modality. T1:0, T2:1, PD:2")

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="Which gpus to use?")

    parser.add_argument("--trained_model",
                        type=str,
                        default="./weights/trained_model.pth.tar",
                        help="trained_model")

    parser.add_argument("--save_dir",
                        type=str,
                        dest="save_dir",
                        default="./eval_output/",
                        help="Path to the folder where data is saved")

    parser.add_argument("--eval_dir",
                        type=str,
                        dest="eval_dir",
                        default="./data/centered_IXI/",
                        help="Path to the training data")

    args = parser.parse_args()
    return args


def run(moving_loader,
        fixed_dataset,
        model,
        optimizer,
        epoch,
        mode,
        args,
        PATH,
        steps_per_epoch):
    u1 = model
    u1.eval()
    for j, data in tqdm(enumerate(moving_loader)):
        n_template = np.random.randint(0, high=N)
        # random subject to use as template/fixed/target image

        """Target"""
        x_tg = fixed_dataset[n_template]['mri'][tio.DATA].unsqueeze(0).float()
        xtg_mask = fixed_dataset[n_template]['mask'][tio.DATA].unsqueeze(0).float()

        """Moving"""
        x = data['mri'][tio.DATA].float()
        x_mask = data['mask'][tio.DATA].float()

        """Augment"""
        Ma = aug.affine_matrix(x.size(),
                               s=0.2,
                               o=0.2,
                               a=3.1416,
                               z=0.1,
                               cuda=False)

        grid = F.affine_grid(torch.inverse(Ma)[:, :3, :],
                             x.size(),
                             align_corners=False)

        x = F.grid_sample(x,
                          grid=grid,
                          mode='bilinear',
                          padding_mode='border',
                          align_corners=False)

        x_mask = F.grid_sample(x_mask,
                               grid=grid,
                               mode='nearest',
                               padding_mode='border',
                               align_corners=False)

        """Predict"""
        x = x_mask.cuda() * x.cuda()
        x_tg = xtg_mask.cuda() * x_tg.cuda()

        moving_kp = u1(F.avg_pool3d(x, 2, 2))
        target_kp = u1(F.avg_pool3d(x_tg, 2, 2))

        """Close Form Affine Matrix"""
        affine_matrix = rt.close_form_affine(moving_kp, target_kp)
        inv_matrix = torch.zeros(x.size(0), 4, 4).cuda() if moving_kp.is_cuda else torch.zeros(x.size(0), 4, 4)
        inv_matrix[:, :3, :4] = affine_matrix
        inv_matrix[:, 3, 3] = 1
        inv_matrix = torch.inverse(inv_matrix)[:, :3, :]

        # Interpolation
        grid = F.affine_grid(inv_matrix,
                             x.size(),
                             align_corners=False)

        x_aligned = F.grid_sample(x,
                                  grid=grid,
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False)

        state = {'moving': x.detach().cpu().squeeze(),
                 'target': x_tg.detach().cpu().squeeze(),
                 'aligned': x_aligned.detach().cpu().squeeze()}
        torch.save(state, PATH + 'imagePair{}.pth.tar'.format(j))

        if PATH is not None:
            # Project the keypoints to 2D plane to visualize the result        
            size = 256
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
                        epoch='imagePair',
                        suffix=str(j),
                        image_idx=0,
                        PATH=PATH,
                        vmin=None,
                        vmax=None)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    from functions import visualization as vis

    eval_args = eval_parser()

    """Select GPU"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = eval_args.gpus
    print('Number of GPUs {}'.format(torch.cuda.device_count()))

    summary = torch.load(eval_args.trained_model)
    args = summary['args']

    """Load Data"""
    directory = eval_args.eval_dir
    N = len(os.listdir(directory + '/T1/'))  # use all subject in the folder
    # Note that in our experiment we use start_end=[0,427] for training
    # start_end=[427,477] for validation
    # start_end=[477,577] for testing
    # these are the range of indeces use for training, validation and testing, respectively
    # feel free to change how to partition the dataset based on your experiment

    transform = Lambda(lambda x: x.permute(0, 1, 3, 2))
    _, moving_loader = loader.create(directory,
                                     start_end=[0, N],
                                     modality=['T1', 'T2', 'PD'][eval_args.input_mod],
                                     transform=transform,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=0,
                                     seed=23)

    fixed_dataset, fixed_loader = loader.create(directory,
                                                start_end=[0, N],
                                                modality=['T1', 'T2', 'PD'][eval_args.target_mod],
                                                transform=transform,
                                                batch_size=1,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=0,
                                                seed=23)
    """Model"""
    # Load model + weights
    u1 = m.KeyMorph(1, args.out_dim, args.norm_type)
    u1 = torch.nn.DataParallel(u1)
    u1.cuda()
    u1.load_state_dict(summary['u1'])
    del summary

    params = list(u1.parameters())

    optimizer = torch.optim.Adam(params,
                                 lr=args.lr)

    input_mod = ['T1', 'T2', 'PD'][eval_args.input_mod]
    target_mod = ['T1', 'T2', 'PD'][eval_args.target_mod]

    arguments = ('[eval]'
                 + '_input' + input_mod
                 + '_target' + target_mod)

    PATH = eval_args.save_dir + arguments + '/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    '''Eval'''
    with torch.no_grad():
        stat = run(moving_loader=moving_loader,
                   fixed_dataset=fixed_dataset,
                   model=u1,
                   optimizer=None,
                   epoch=None,
                   mode='eval',
                   args=args,
                   PATH=PATH,
                   steps_per_epoch=None)
