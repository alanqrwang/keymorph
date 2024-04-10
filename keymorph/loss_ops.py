import torch
import torch.nn.functional as F
import scipy
from scipy.ndimage import morphology
import numpy as np
import nibabel as nib


class MSELoss(torch.nn.Module):
    """MSE loss."""

    def forward(self, pred, target):
        return F.mse_loss(pred, target)


class DiceLoss(torch.nn.Module):
    """Dice loss (lower is better).

    Supports 2d or 3d inputs.
    Supports hard dice or soft dice.
    If soft dice, returns scalar dice loss for entire slice/volume.
    If hard dice, returns:
        total_avg: scalar dice loss for entire slice/volume ()
        regions_avg: dice loss per region (n_ch)
        ind_avg: dice loss per region per pixel (bs, n_ch)

    """

    def __init__(self, hard=False, return_regions=False):
        super(DiceLoss, self).__init__()
        self.hard = hard
        self.return_regions = return_regions

    def forward(self, pred, target, ign_first_ch=False):
        eps = 1
        assert pred.size() == target.size(), "Input and target are different dim"

        if len(target.size()) == 4:
            n, c, _, _ = target.size()
        if len(target.size()) == 5:
            n, c, _, _, _ = target.size()

        target = target.contiguous().view(n, c, -1)
        pred = pred.contiguous().view(n, c, -1)

        if self.hard:  # hard Dice
            pred_onehot = torch.zeros_like(pred)
            pred = torch.argmax(pred, dim=1, keepdim=True)
            pred = torch.scatter(pred_onehot, 1, pred, 1.0)
        if ign_first_ch:
            target = target[:, 1:, :]
            pred = pred[:, 1:, :]

        num = torch.sum(2 * (target * pred), 2) + eps
        den = (pred * pred).sum(2) + (target * target).sum(2) + eps
        dice_loss = 1 - num / den
        total_avg = torch.mean(dice_loss)
        regions_avg = torch.mean(dice_loss, 0)

        if self.return_regions:
            return regions_avg
        else:
            return total_avg


def fast_dice(x, y):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on
    :return: numpy array with Dice scores in the same order as labels.
    """

    x = x.argmax(1)
    y = y.argmax(1)
    labels = np.unique(np.concatenate([np.unique(x), np.unique(y)]))
    assert (
        x.shape == y.shape
    ), "both inputs should have same size, had {} and {}".format(x.shape, y.shape)

    if len(labels) > 1:
        # sort labels
        labels_sorted = np.sort(labels)

        # build bins for histograms
        label_edges = np.sort(
            np.concatenate([labels_sorted - 0.1, labels_sorted + 0.1])
        )
        label_edges = np.insert(
            label_edges,
            [0, len(label_edges)],
            [labels_sorted[0] - 0.1, labels_sorted[-1] + 0.1],
        )

        # compute Dice and re-arrange scores in initial order
        hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)[0]
        idx = np.arange(start=1, stop=2 * len(labels_sorted), step=2)
        dice_score = (
            2 * np.diag(hst)[idx] / (np.sum(hst, 0)[idx] + np.sum(hst, 1)[idx] + 1e-5)
        )
        dice_score = dice_score[np.searchsorted(labels_sorted, labels)]

    else:
        dice_score = dice(x == labels[0], y == labels[0])

    return np.mean(dice_score)  # Remove mean to get region-level scores


def dice(x, y):
    """Implementation of dice scores for 0/1 numpy array"""
    return 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))


def _check_type(t):
    if isinstance(t, torch.Tensor):
        t = t.cpu().detach().numpy()
    return t


# Hausdorff Distance
def _surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(bool))
    input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = (
        input_1.astype(int) - morphology.binary_erosion(input_1, conn).astype(int)
    ).astype(bool)
    Sprime = (
        input_2.astype(int) - morphology.binary_erosion(input_2, conn).astype(int)
    ).astype(bool)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds


def hausdorff_distance(test_seg, gt_seg):
    """Computes Hausdorff distance on brain surface.

    Assumes segmentation maps are one-hot and first channel is background.

    Args:
        test_seg: Test segmentation map (bs, n_ch, l, w, h)
        gt_seg: Ground truth segmentation map (bs, n_ch, l, w, h)
    """
    test_seg = _check_type(test_seg)
    gt_seg = _check_type(gt_seg)

    hd = 0
    for i in range(len(test_seg)):
        hd += _surfd(test_seg[i, 0], gt_seg[i, 0], [1.25, 1.25, 10], 1).max()
    return hd / len(test_seg)


# Jacobian determinant
def _jacobian_determinant(disp):
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradz_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradz, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradz, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grady_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], grady, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], grady, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    gradx_disp = np.stack(
        [
            scipy.ndimage.correlate(
                disp[:, 0, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 1, :, :, :], gradx, mode="constant", cval=0.0
            ),
            scipy.ndimage.correlate(
                disp[:, 2, :, :, :], gradx, mode="constant", cval=0.0
            ),
        ],
        axis=1,
    )

    grad_disp = np.concatenate([gradz_disp, grady_disp, gradx_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    return jacdet


def jdstd(disp):
    disp = _check_type(disp)
    jd = _jacobian_determinant(disp)
    return jd.std()


def jdlessthan0(disp, as_percentage=False):
    disp = _check_type(disp)
    jd = _jacobian_determinant(disp)
    if as_percentage:
        return np.count_nonzero(jd <= 0) / len(jd.flatten())
    return np.count_nonzero(jd <= 0)


class LC2(torch.nn.Module):
    def __init__(self, radiuses=(3, 5, 7)):
        super(LC2, self).__init__()
        self.radiuses = radiuses
        self.f = torch.zeros(3, 1, 3, 3, 3)
        self.f[0, 0, 1, 1, 0] = 1
        self.f[0, 0, 1, 1, 2] = -1
        self.f[1, 0, 1, 0, 1] = 1
        self.f[1, 0, 1, 2, 1] = -1
        self.f[2, 0, 0, 1, 1] = 1
        self.f[2, 0, 2, 1, 1] = -1

    def forward(self, us, mr):
        s = self.run(us, mr, self.radiuses[0])
        for r in self.radiuses[1:]:
            s += self.run(us, mr, r)
        return s / len(self.radiuses)

    def run(self, us, mr, radius=9, alpha=1e-3, beta=1e-2):
        us = us.squeeze(1)
        mr = mr.squeeze(1)
        assert us.shape == mr.shape
        assert us.shape[1] == us.shape[2] == us.shape[3]
        assert us.shape[1] % 2 == 1, "Input must be odd size"

        bs = mr.size(0)
        pad = (mr.size(1) - (2 * radius + 1)) // 2
        count = (2 * radius + 1) ** 3

        self.f = self.f.to(mr.device)

        grad = torch.norm(F.conv3d(mr.unsqueeze(1), self.f, padding=1), dim=1)

        A = torch.ones(bs, 3, count, device=mr.device)
        A[:, 0] = mr[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        A[:, 1] = grad[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        b = us[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)

        C = (
            torch.einsum("bip,bjp->bij", A, A) / count
            + torch.eye(3, device=mr.device).unsqueeze(0) * alpha
        )
        Atb = torch.einsum("bip,bp->bi", A, b) / count
        coeff = torch.linalg.solve(C, Atb)
        var = torch.mean(b**2, dim=1) - torch.mean(b, dim=1) ** 2
        dist = (
            torch.mean(b**2, dim=1)
            + torch.einsum("bi,bj,bij->b", coeff, coeff, C)
            - 2 * torch.einsum("bi,bi->b", coeff, Atb)
        )
        sym = (var - dist) / var.clamp_min(beta)

        return sym.clamp(0, 1)


class ImageLC2(torch.nn.Module):
    def __init__(self, patch_size=51, radiuses=(5,), reduction="mean"):
        super(ImageLC2, self).__init__()
        self.patch_size = patch_size
        self.radii = radiuses
        assert reduction in ["mean", None]
        self.reduction = reduction
        self.f = torch.zeros(3, 1, 3, 3, 3)
        self.f[0, 0, 1, 1, 0] = 1
        self.f[0, 0, 1, 1, 2] = -1
        self.f[1, 0, 1, 0, 1] = 1
        self.f[1, 0, 1, 2, 1] = -1
        self.f[2, 0, 0, 1, 1] = 1
        self.f[2, 0, 2, 1, 1] = -1

    @staticmethod
    def patch2batch(x, size, stride):
        """Converts image x into patches, then reshapes into batch of patches"""
        nch = x.shape[1]
        if len(x.shape) == 4:
            patches = x.unfold(2, size, stride).unfold(3, size, stride)
            return patches.reshape(-1, nch, size, size)
        else:
            patches = (
                x.unfold(2, size, stride)
                .unfold(3, size, stride)
                .unfold(4, size, stride)
            )
            return patches.reshape(-1, nch, size, size, size)

    def forward(self, us, mr):
        assert (
            us.shape == mr.shape
        ), f"Input and target have different shapes, {us.shape} vs {mr.shape}"
        assert (
            us.shape[-1] == us.shape[-2] == us.shape[-3]
        ), f"Dimensions must be equal, currently {us.shape}"
        assert us.shape[1] % 2 == 1, f"Input must be odd size, currently {us.shape}"

        # Convert to batch of patches
        r = self.radii[0]
        us_patch = self.patch2batch(us, self.patch_size, self.patch_size)
        mr_patch = self.patch2batch(mr, self.patch_size, self.patch_size)
        s = self.run(us_patch, mr_patch, r)
        for r in self.radii[1:]:
            us_patch = self.patch2batch(us, self.patch_size, self.patch_size)
            mr_patch = self.patch2batch(mr, self.patch_size, self.patch_size)
            s += self.run(us_patch, mr_patch, r)

        s = s / len(self.radii)
        if self.reduction == "mean":
            return s.mean()
        else:
            return s

    def run(self, us, mr, radius=9, alpha=1e-3, beta=1e-2):
        us = us.squeeze(1)
        mr = mr.squeeze(1)

        bs = mr.size(0)
        pad = (mr.size(1) - (2 * radius + 1)) // 2
        count = (2 * radius + 1) ** 3

        self.f = self.f.to(mr.device)

        grad = torch.norm(F.conv3d(mr.unsqueeze(1), self.f, padding=1), dim=1)

        A = torch.ones(bs, 3, count, device=mr.device)
        A[:, 0] = mr[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        A[:, 1] = grad[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)
        b = us[:, pad:-pad, pad:-pad, pad:-pad].reshape(bs, -1)

        C = (
            torch.einsum("bip,bjp->bij", A, A) / count
            + torch.eye(3, device=mr.device).unsqueeze(0) * alpha
        )
        Atb = torch.einsum("bip,bp->bi", A, b) / count
        coeff = torch.linalg.solve(C, Atb)
        var = torch.mean(b**2, dim=1) - torch.mean(b, dim=1) ** 2
        dist = (
            torch.mean(b**2, dim=1)
            + torch.einsum("bi,bj,bij->b", coeff, coeff, C)
            - 2 * torch.einsum("bi,bi->b", coeff, Atb)
        )
        sym = (var - dist) / var.clamp_min(beta)

        return sym.clamp(0, 1)


# class LesionPenalty(torch.nn.Module):
#     def forward(self, weights, points_f, points_m, lesion_mask_f, lesion_mask_m):
#         ind_in_mask_f = ind_in_lesion(points_f, lesion_mask_f)
#         ind_in_mask_m = ind_in_lesion(points_m, lesion_mask_m)

#         gt = torch.ones_like(weights)
#         gt[ind_in_mask_f] = 0
#         gt[ind_in_mask_m] = 0

#         return F.mse_loss(gt, weights)


def _load_file(path):
    if path.endswith(".nii") or path.endswith(".nii.gz"):
        return torch.tensor(nib.load(path).get_fdata())
    elif path.endswith(".npy"):
        return torch.tensor(np.load(path))
    else:
        raise ValueError("File format not supported")


class _AvgPairwiseLoss(torch.nn.Module):
    """Pairwise loss."""

    def __init__(self, metric_fn):
        super().__init__()
        self.metric_fn = metric_fn

    def forward(self, batch_of_imgs):
        loss = 0
        num = 0
        for i in range(len(batch_of_imgs)):
            for j in range(i + 1, len(batch_of_imgs)):
                if isinstance(batch_of_imgs[0], str):
                    img1 = _load_file(batch_of_imgs[i])
                    img2 = _load_file(batch_of_imgs[j])
                else:
                    img1 = batch_of_imgs[i : i + 1]
                    img2 = batch_of_imgs[j : j + 1]
                loss += self.metric_fn(img1, img2)
                num += 1
        return loss / num


class MSEPairwiseLoss(_AvgPairwiseLoss):
    """MSE pairwise loss."""

    def __init__(self):
        super().__init__(MSELoss().forward)


class SoftDicePairwiseLoss(_AvgPairwiseLoss):
    """Soft Dice pairwise loss."""

    def __init__(self):
        super().__init__(DiceLoss().forward)


class HardDicePairwiseLoss(_AvgPairwiseLoss):
    """Hard Dice pairwise loss."""

    def __init__(self):
        super().__init__(DiceLoss(hard=True).forward)


class HausdorffPairwiseLoss(_AvgPairwiseLoss):
    """Hausdorff pairwise loss."""

    def __init__(self):
        super().__init__(hausdorff_distance)


class _AvgGridMetric(torch.nn.Module):
    """Aggregated average metric for grids."""

    def __init__(self, metric_fn):
        super().__init__()
        self.metric_fn = metric_fn

    def forward(self, batch_of_grids):
        tot_jdstd = 0
        for i in range(len(batch_of_grids)):
            if isinstance(batch_of_grids[i], str):
                grid = _load_file(batch_of_grids[i])
            else:
                grid = batch_of_grids[i : i + 1]
            grid_permute = grid.permute(0, 4, 1, 2, 3)
            tot_jdstd += self.metric_fn(grid_permute)
        return tot_jdstd / len(batch_of_grids)


class AvgJDStd(_AvgGridMetric):
    """Soft Dice pairwise loss."""

    def __init__(self):
        super().__init__(jdstd)


class AvgJDLessThan0(_AvgGridMetric):
    """Soft Dice pairwise loss."""

    def __init__(self):
        super().__init__(jdlessthan0)


class MultipleAvgSegPairwiseMetric(torch.nn.Module):
    """Evaluate multiple pairwise losses on a batch of images, 
    so that we don't need to load the images into memory multiple times."""

    def __init__(self):
        super().__init__()
        self.name2fn = {
            "dice": fast_dice,
            "harddice": DiceLoss(hard=True).forward,
            "harddiceroi": DiceLoss(hard=True, return_regions=True).forward,
            "softdice": DiceLoss().forward,
            "hausd": hausdorff_distance,
        }

    def forward(self, batch_of_imgs, fn_names):
        res = {name: 0 for name in fn_names}
        num = 0
        for i in range(len(batch_of_imgs)):
            for j in range(i + 1, len(batch_of_imgs)):
                if isinstance(batch_of_imgs[0], str):
                    img1 = _load_file(batch_of_imgs[i])
                    img2 = _load_file(batch_of_imgs[j])
                else:
                    img1 = batch_of_imgs[i : i + 1]
                    img2 = batch_of_imgs[j : j + 1]
                for name in fn_names:
                    res[name] += self.name2fn[name](img1, img2)
                num += 1
        return {name: res[name] / num for name in fn_names}


class MultipleAvgGridMetric(torch.nn.Module):
    """Evaluate multiple grid metrics on a batch of grids, mostly
    so that we don't need to load them into memory multiple times."""

    def __init__(self):
        super().__init__()
        self.name2fn = {
            "jdstd": jdstd,
            "jdlessthan0": jdlessthan0,
        }

    def forward(self, batch_of_grids, fn_names):
        res = {name: 0 for name in fn_names}
        for i in range(len(batch_of_grids)):
            if isinstance(batch_of_grids[i], str):
                grid = _load_file(batch_of_grids[i])
            else:
                grid = batch_of_grids[i : i + 1]
            grid_permute = grid.permute(0, 4, 1, 2, 3)
            for name in fn_names:
                res[name] += self.name2fn[name](grid_permute)
        return {name: res[name] / len(batch_of_grids) for name in fn_names}
