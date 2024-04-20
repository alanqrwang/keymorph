import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint


class MatrixKeypointAligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def get_matrix(self, p1, p2, w=None):
        pass

    def forward(self, *args, **kwargs):
        return self.grid_from_points(*args, **kwargs)

    def grid_from_points(
        self,
        points_m,
        points_f,
        grid_shape,
        weights=None,
        lmbda=None,
        compute_on_subgrids=False,
    ):
        # Note we flip the order of the points here
        matrix = self.get_matrix(points_f, points_m, w=weights)
        grid = F.affine_grid(matrix, grid_shape, align_corners=False)
        return grid

    def deform_points(self, points, matrix):
        square_mat = torch.zeros(len(points), self.dim + 1, self.dim + 1).to(
            points.device
        )
        square_mat[:, : self.dim, : self.dim + 1] = matrix
        square_mat[:, -1, -1] = 1
        batch_size, num_points, _ = points.shape

        points = torch.cat(
            (points, torch.ones(batch_size, num_points, 1).to(points.device)), dim=-1
        )
        warp_points = torch.bmm(square_mat[:, :3, :], points.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        return warp_points

    def points_from_points(
        self, moving_points, fixed_points, points, weights=None, **kwargs
    ):
        affine_matrix = self.get_matrix(moving_points, fixed_points, w=weights)
        square_mat = torch.zeros(len(points), self.dim + 1, self.dim + 1).to(
            moving_points.device
        )
        square_mat[:, : self.dim, : self.dim + 1] = affine_matrix
        square_mat[:, -1, -1] = 1
        batch_size, num_points, _ = points.shape

        points = torch.cat(
            (points, torch.ones(batch_size, num_points, 1).to(moving_points.device)),
            dim=-1,
        )
        warped_points = torch.bmm(
            square_mat[:, :3, :], points.permute(0, 2, 1)
        ).permute(0, 2, 1)
        return warped_points


class RigidKeypointAligner(MatrixKeypointAligner):
    def __init__(self, dim):
        super().__init__(dim)

    def get_matrix(self, p1, p2, w=None):
        """
        Find R and T which is the solution to argmin_{R, T} \sum_i ||p2_i - (R * p1_i + T)||_2
        See https://ieeexplore.ieee.org/document/4767965


        Args:
          x, y: [n_batch, n_points, dim]
          w: [n_batch, n_points]
        Returns:
          A: [n_batch, dim, dim+1]
        """
        # Take transpose as columns should be the points
        p1 = p1.permute(0, 2, 1)
        p2 = p2.permute(0, 2, 1)

        # Calculate centroids
        if w is not None:
            # find weighed mean column wise
            p1_c = torch.sum(p1 * w, axis=2, keepdims=True)
            p2_c = torch.sum(p2 * w, axis=2, keepdims=True)
        else:
            p1_c = torch.mean(p1, axis=2, keepdim=True)
            p2_c = torch.mean(p2, axis=2, keepdim=True)

        # Subtract centroids
        q1 = p1 - p1_c
        q2 = p2 - p2_c

        if w is not None:
            q1 = q1 * w
            q2 = q2 * w

        # Calculate covariance matrix
        H = torch.bmm(q1, q2.transpose(1, 2))

        # Calculate singular value decomposition (SVD)
        U, _, Vt = torch.linalg.svd(H)  # the SVD of linalg gives you Vt
        Ut = U.transpose(1, 2)
        V = Vt.transpose(1, 2)
        R = torch.bmm(V, Ut)

        # assert torch.allclose(
        #     torch.linalg.det(R), torch.tensor(1.0)
        # ), f"Rotation matrix of N-point registration not 1, see paper Arun et al., det={torch.linalg.det(R)}"

        # special reflection case
        dets = torch.det(R)
        dets = torch.unsqueeze(dets, -1)
        dets = torch.stack([torch.ones_like(dets), torch.ones_like(dets), dets], axis=1)
        dets = torch.cat([dets, dets, dets], axis=2)
        V = V * torch.sign(dets)

        # Calculate rotation matrix
        R = torch.bmm(V, Ut)

        # Calculate translation matrix
        T = p2_c - torch.bmm(R, p1_c)

        # Create augmented matrix
        aug_mat = torch.cat([R, T], axis=-1)
        return aug_mat


class AffineKeypointAligner(MatrixKeypointAligner):
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim

    def get_matrix(self, x, y, w=None):
        """
        Find A which is the solution to argmin_A \sum_i ||y_i - Ax_i||_2 = argmin_A ||Ax - y||_F
        Computes the closed-form affine equation: A = y x^T (x x^T)^(-1).

        If w provided, solves the weighted affine equation:
          A = y diag(w) x^T  (x diag(w) x^T)^(-1).
          See https://www.wikiwand.com/en/Weighted_least_squares.

        Args:
          x, y: [n_batch, n_points, dim]
          w: [n_batch, n_points]
        Returns:
          A: [n_batch, dim, dim+1]
        """
        # Take transpose as columns should be the points
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        if w is not None:
            w = torch.diag_embed(w)

        # Convert y to homogenous coordinates
        one = torch.ones(x.shape[0], 1, x.shape[2]).float().to(x.device)
        x = torch.cat([x, one], 1)

        if w is not None:
            out = torch.bmm(x, w)
            out = torch.bmm(out, torch.transpose(x, -2, -1))
        else:
            out = torch.bmm(x, torch.transpose(x, -2, -1))
        inv = torch.inverse(out)
        if w is not None:
            out = torch.bmm(w, torch.transpose(x, -2, -1))
            out = torch.bmm(out, inv)
        else:
            out = torch.bmm(torch.transpose(x, -2, -1), inv)
        out = torch.bmm(y, out)
        return out


class TPS(nn.Module):
    """See https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/numpy.py"""

    def __init__(self, dim, num_subgrids=4, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_subgrids = num_subgrids
        self.use_checkpoint = use_checkpoint

    def fit(self, c, lmbda, w=None):
        """Assumes last dimension of c contains target points.

          Set up and solve linear system:
            [K + lmbda*I   P] [w] = [v]
            [        P^T   0] [a]   [0]

          If w is provided, solve weighted TPS:
            [K + lmbda*1/diag(w)   P] [w] = [v]
            [                P^T   0] [a]   [0]

          See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=929618&tag=1, Eq. (8)
        Args:
          c: control points and target point (bs, T, d+1)
          lmbda: Lambda values per batch (bs)
        """
        device = c.device
        bs, T = c.shape[0], c.shape[1]
        ctrl, tgt = c[:, :, : self.dim], c[:, :, -1]

        # Build K matrix
        U = TPS.u(TPS.d(ctrl, ctrl))
        if w is not None:
            w = torch.diag_embed(w)
            K = U + torch.reciprocal(w + 1e-6) * lmbda.view(
                bs, 1, 1
            )  # w are weights, TPS expects variances
        else:
            I = torch.eye(T).repeat(bs, 1, 1).float().to(device)
            K = U + I * lmbda.view(bs, 1, 1)

        # Build P matrix
        P = torch.ones((bs, T, self.dim + 1)).float()
        P[:, :, 1:] = ctrl

        # Build v vector
        v = torch.zeros(bs, T + self.dim + 1).float()
        v[:, :T] = tgt

        A = torch.zeros((bs, T + self.dim + 1, T + self.dim + 1)).float()
        A[:, :T, :T] = K
        A[:, :T, -(self.dim + 1) :] = P
        A[:, -(self.dim + 1) :, :T] = P.transpose(1, 2)

        return torch.linalg.solve(A, v)

    @staticmethod
    def d(a, b):
        """Compute pair-wise distances between points.

        Args:
          a: (bs, num_points, d)
          b: (bs, num_points, d)
        Returns:
          dist: (bs, num_points, num_points)
        """
        return torch.sqrt(
            torch.square(a[:, :, None, :] - b[:, None, :, :]).sum(-1) + 1e-6
        )

    @staticmethod
    def u(r):
        """Compute radial basis function."""
        return r**2 * torch.log(r + 1e-6)

    def tps_theta_from_points(self, c_src, c_dst, lmbda, weights=None):
        """
        Args:
          c_src: (bs, T, dim)
          c_dst: (bs, T, dim)
          lmbda: (bs)
        """
        device = c_src.device

        cx = torch.cat((c_src, c_dst[..., 0:1]), dim=-1)
        cy = torch.cat((c_src, c_dst[..., 1:2]), dim=-1)
        if self.dim == 3:
            cz = torch.cat((c_src, c_dst[..., 2:3]), dim=-1)

        theta_dx = self.fit(cx, lmbda, w=weights).to(device)
        theta_dy = self.fit(cy, lmbda, w=weights).to(device)
        if self.dim == 3:
            theta_dz = self.fit(cz, lmbda, w=weights).to(device)

        if self.dim == 3:
            return torch.stack((theta_dx, theta_dy, theta_dz), -1)
        else:
            return torch.stack((theta_dx, theta_dy), -1)

    def tps(self, theta, ctrl, grid):
        """Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
        The TPS surface is a minimum bend interpolation surface defined by a set of control points.
        The function value for a x,y location is given by

          TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])

        This method computes the TPS value for multiple batches over multiple grid locations for 2
        surfaces in one go.

        Params
        ------
        theta: Nx(T+3)xd tensor, or Nx(T+2)xd tensor
          Batch size N, T+3 model parameters for T control points in dx and dy.
        ctrl: NxTxd tensor
          T control points in normalized image coordinates [0..1]
        grid: NxHxWx(d+1) tensor
          Grid locations to evaluate with homogeneous 1 in first coordinate.

        Returns
        -------
        z: NxHxWxd tensor
          Function values at each grid location in dx and dy.
        """

        if len(grid.shape) == 4:
            N, H, W, _ = grid.size()
            x = grid[..., 1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
        else:
            N, D, H, W, _ = grid.size()
            x = grid[..., 1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            )

        T = ctrl.shape[1]

        x = torch.sqrt((x**2).sum(-1))
        x = TPS.u(x)

        w, a = theta[:, : -(self.dim + 1), :], theta[:, -(self.dim + 1) :, :]

        # x is NxHxWxT
        # b contains dot product of each kernel weight and U(r)
        b = torch.bmm(x.view(N, -1, T), w)
        if len(grid.shape) == 4:
            b = b.view(N, H, W, self.dim)
        else:
            b = b.view(N, D, H, W, self.dim)

        # b is NxHxWxd
        # z contains dot product of each affine term and polynomial terms.
        z = torch.bmm(grid.reshape(N, -1, self.dim + 1), a)
        if len(grid.shape) == 4:
            z = z.view(N, H, W, self.dim) + b
        else:
            z = z.view(N, D, H, W, self.dim) + b
        return z

    def tps_grid(self, theta, ctrl, size, compute_on_subgrids=False):
        """Compute a thin-plate-spline grid from parameters for sampling.

        Params
        ------
        theta: Nx(T+3)xd tensor
          Batch size N, T+3 model parameters for T control points in dx and dy.
        ctrl: NxTxd tensor, or Txdim tensor
          T control points in normalized image coordinates [0..1]
        size: tuple
          Output grid size as NxCxHxW. C unused. This defines the output image
          size when sampling.
        compute_on_subgrids: If true, compute the TPS grid on several subgrids
            for memory efficiency. This is useful when the grid is large, but only
            works for inference time. At training, gradients need to be persisted
            for the entire grid, so computing on subgrids makes no difference.

        Returns
        -------
        grid : NxHxWxd tensor
          Grid suitable for sampling in pytorch containing source image
          locations for each output pixel.
        """
        device = theta.device
        if len(size) == 4:
            N, _, H, W = size
            grid_shape = (N, H, W, self.dim + 1)
        else:
            N, _, D, H, W = size
            grid_shape = (N, D, H, W, self.dim + 1)
        grid = self.uniform_grid(grid_shape).to(device)

        if compute_on_subgrids:
            output_shape = list(grid.shape)
            output_shape[-1] -= 1
            z = torch.zeros(output_shape).to(device)
            size_x, size_y, size_z = grid.shape[1:-1]
            assert size_x % self.num_subgrids == 0
            assert size_y % self.num_subgrids == 0
            assert size_z % self.num_subgrids == 0
            subsize_x, subsize_y, subsize_z = (
                size_x // self.num_subgrids,
                size_y // self.num_subgrids,
                size_z // self.num_subgrids,
            )
            for i in range(self.num_subgrids):
                for j in range(self.num_subgrids):
                    for k in range(self.num_subgrids):
                        subgrid = grid[
                            :,
                            i * subsize_x : (i + 1) * subsize_x,
                            j * subsize_y : (j + 1) * subsize_y,
                            k * subsize_z : (k + 1) * subsize_z,
                            :,
                        ]
                        if self.use_checkpoint:
                            z[
                                :,
                                i * subsize_x : (i + 1) * subsize_x,
                                j * subsize_y : (j + 1) * subsize_y,
                                k * subsize_z : (k + 1) * subsize_z,
                                :,
                            ] = checkpoint.checkpoint(self.tps, theta, ctrl, subgrid)
                        else:
                            z[
                                :,
                                i * subsize_x : (i + 1) * subsize_x,
                                j * subsize_y : (j + 1) * subsize_y,
                                k * subsize_z : (k + 1) * subsize_z,
                                :,
                            ] = self.tps(theta, ctrl, subgrid)
        else:
            z = self.tps(theta, ctrl, grid)
        return z

    def uniform_grid(self, shape):
        """Uniform grid coordinates.

        Params
        ------
        shape : tuple
            NxHxWx3 defining the batch size, height and width dimension of the grid.
            3 is for the number of dimensions (2) plus 1 for the homogeneous coordinate.
        Returns
        -------
        grid: HxWx3 tensor
            Grid coordinates over [-1,1] normalized image range.
            Homogenous coordinate in first coordinate position.
            After that, the second coordinate varies first, then
            the third coordinate varies, then (optionally) the
            fourth coordinate varies.
        """

        if self.dim == 2:
            _, H, W, _ = shape
        else:
            _, D, H, W, _ = shape
        grid = torch.zeros(shape)

        grid[..., 0] = 1.0
        grid[..., 1] = torch.linspace(-1, 1, W)
        grid[..., 2] = torch.linspace(-1, 1, H).unsqueeze(-1)
        if grid.shape[-1] == 4:
            grid[..., 3] = torch.linspace(-1, 1, D).unsqueeze(-1).unsqueeze(-1)
        return grid

    def forward(self, *args, **kwargs):
        return self.grid_from_points(*args, **kwargs)

    def grid_from_points(
        self,
        points_m,
        points_f,
        grid_shape,
        weights=None,
        lmbda=None,
        compute_on_subgrids=False,
    ):
        # Note we flip the order of the points here
        if self.use_checkpoint:
            theta = checkpoint.checkpoint(
                self.tps_theta_from_points, points_f, points_m, lmbda, weights
            )
        else:
            theta = self.tps_theta_from_points(
                points_f, points_m, lmbda, weights=weights
            )
        return self.tps_grid(
            theta, points_f, grid_shape, compute_on_subgrids=compute_on_subgrids
        )

    def deform_points(self, theta, ctrl, points):
        weights, affine = theta[:, : -(self.dim + 1), :], theta[:, -(self.dim + 1) :, :]
        N, T, _ = ctrl.shape
        U = TPS.u(TPS.d(ctrl, points))

        P = torch.ones((N, points.shape[1], self.dim + 1)).float().to(theta.device)
        P[:, :, 1:] = points[:, :, : self.dim]

        # U is NxHxWxT
        b = torch.bmm(U.transpose(1, 2), weights)
        z = torch.bmm(P.view(N, -1, self.dim + 1), affine)
        return z + b

    def points_from_points(
        self, ctl_points, tgt_points, points, weights=None, **kwargs
    ):
        lmbda = kwargs["lmbda"]
        theta = self.tps_theta_from_points(
            ctl_points, tgt_points, lmbda, weights=weights
        )
        return self.deform_points(theta, ctl_points, points)


class ApproximateTPS(TPS):
    """Method 2 from ``Approximate TPS Mappings'' by Donato and Belongie"""

    def fit(self, c, lmbda, subsample_indices, w=None):
        """Assumes last dimension of c contains target points.

          Set up and solve linear system:
            [K + lmbda*I   P] [w] = [v]
            [        P^T   0] [a]   [0]

          If w is provided, solve weighted TPS:
            [K + lmbda*1/diag(w)   P] [w] = [v]
            [                P^T   0] [a]   [0]

          See https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=929618&tag=1, Eq. (8)
        Args:
          c: control points and target point (bs, T, d+1)
          lmbda: Lambda values per batch (bs)
        """
        device = c.device
        bs, T = c.shape[0], c.shape[1]
        ctrl, tgt = c[:, :, : self.dim], c[:, :, -1]
        num_subsample = len(subsample_indices)
        print(num_subsample)

        # Build K matrix
        print("ctrl shape", ctrl.shape)
        U = TPS.u(TPS.d(ctrl, ctrl[:, subsample_indices]))
        print("U shape", U.shape)
        if w is not None:
            w = torch.diag_embed(w)
            w = w[:, :, subsample_indices]
            print("w shape", w.shape)
            K = U + torch.reciprocal(w + 1e-6) * lmbda.view(
                bs, 1, 1
            )  # w are weights, TPS expects variances
        else:
            I = torch.eye(T).repeat(bs, 1, 1).float().to(device)
            I = I[:, :, subsample_indices]
            K = U + I * lmbda.view(bs, 1, 1)

        # Build P matrix
        P = torch.ones((bs, T, self.dim + 1)).float()
        P[:, :, 1:] = ctrl
        P_tilde = P[:, subsample_indices]
        print("P shapes", P.shape, P_tilde.shape)

        # Build v vector
        v = torch.zeros(bs, T + self.dim + 1).float()
        v[:, :T] = tgt

        A = torch.zeros((bs, T + self.dim + 1, num_subsample + self.dim + 1)).float()
        A[:, :T, :num_subsample] = K
        A[:, :T, -(self.dim + 1) :] = P
        A[:, -(self.dim + 1) :, :num_subsample] = P_tilde.transpose(1, 2)

        return torch.linalg.lstsq(A, v).solution

    def tps_theta_from_points(
        self, c_src, c_dst, lmbda, subsample_indices, weights=None
    ):
        """
        Args:
          c_src: (bs, T, dim)
          c_dst: (bs, T, dim)
          lmbda: (bs)
        """
        device = c_src.device

        cx = torch.cat((c_src, c_dst[..., 0:1]), dim=-1)
        cy = torch.cat((c_src, c_dst[..., 1:2]), dim=-1)
        if self.dim == 3:
            cz = torch.cat((c_src, c_dst[..., 2:3]), dim=-1)

        theta_dx = self.fit(cx, lmbda, subsample_indices, w=weights).to(device)
        theta_dy = self.fit(cy, lmbda, subsample_indices, w=weights).to(device)
        if self.dim == 3:
            theta_dz = self.fit(cz, lmbda, subsample_indices, w=weights).to(device)

        if self.dim == 3:
            return torch.stack((theta_dx, theta_dy, theta_dz), -1)
        else:
            return torch.stack((theta_dx, theta_dy), -1)

    def grid_from_points(
        self,
        points_m,
        points_f,
        grid_shape,
        subsample_indices,
        weights=None,
        compute_on_subgrids=False,
        **kwargs,
    ):
        lmbda = kwargs["lmbda"]

        assert len(subsample_indices) < points_m.shape[1]
        theta = self.tps_theta_from_points(
            points_f,
            points_m,
            lmbda,
            subsample_indices,
            weights=weights,
        )
        return self.tps_grid(
            theta,
            points_f[:, subsample_indices],
            grid_shape,
            compute_on_subgrids=compute_on_subgrids,
        )

    def points_from_points(
        self, ctl_points, tgt_points, points, subsample_indices, weights=None, **kwargs
    ):
        lmbda = kwargs["lmbda"]
        theta = self.tps_theta_from_points(
            ctl_points,
            tgt_points,
            lmbda,
            weights=weights,
            subsample_indices=subsample_indices,
        )
        return self.deform_points(theta, ctl_points[:, subsample_indices], points)
