"""Non-uniform FFT interpolation PyTorch layer."""
import numpy as np
import torch
import torch.nn as nn

from .functional.kbinterp import AdjKbInterpFunction, KbInterpFunction
from .nufft_utils import build_table


class KbInterpForw(nn.Module):
    """Non-uniform FFT forward interpolation PyTorch layer.

    This object interpolates a grid of Fourier data to off-grid locations
    using a Kaiser-Bessel kernel. It is implemented as a PyTorch nn layer.

    Args:
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate to.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in each
            direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points to
            shift for fftshifts.
        table_oversamp (int, default=2^10): Table oversampling factor.
        order (ind, default=0): Order of Kaiser-Bessel kernel. Not currently
            implemented.
        coil_broadcast (boolean, default=False): Whether to broadcast across
            coil dimension. Much faster for many coils, but uses more memory.
    """

    def __init__(self, im_size, grid_size=None, numpoints=6, n_shift=None,
                 table_oversamp=2**10, order=0, coil_broadcast=False, matadj=False):

        super(KbInterpForw, self).__init__()

        self.alpha = 2.34
        self.im_size = im_size
        if grid_size is None:
            self.grid_size = tuple(np.array(self.im_size) * 2)
        else:
            self.grid_size = grid_size
        if n_shift is None:
            self.n_shift = tuple(np.array(self.im_size) // 2)
        else:
            self.n_shift = n_shift
        if numpoints == 6:
            self.numpoints = (6,) * len(self.grid_size)
        elif len(numpoints) != len(self.grid_size):
            self.numpoints = (numpoints,) * len(self.grid_size)
        else:
            self.numpoints = numpoints
        self.order = (0,)
        self.alpha = (2.34 * self.numpoints[0],)
        for i in range(1, len(self.numpoints)):
            self.alpha = self.alpha + (2.34 * self.numpoints[i],)
            self.order = self.order + (0,)
        if table_oversamp == 2**10:
            self.table_oversamp = (table_oversamp,) * len(self.grid_size)
        else:
            self.table_oversamp = table_oversamp
        table = build_table(
            numpoints=self.numpoints,
            table_oversamp=self.table_oversamp,
            grid_size=self.grid_size,
            im_size=self.im_size,
            ndims=len(self.im_size),
            order=self.order,
            alpha=self.alpha
        )
        self.table = table
        self.coil_broadcast = coil_broadcast
        self.matadj = matadj

        # dimension checking
        assert len(self.grid_size) == len(self.im_size)
        assert len(self.n_shift) == len(self.im_size)
        assert len(self.numpoints) == len(self.im_size)
        assert len(self.table) == len(self.im_size)
        assert len(self.table_oversamp) == len(self.im_size)
        assert len(self.alpha) == len(self.im_size)
        assert len(self.order) == len(self.im_size)

        for i, item in enumerate(self.table):
            self.register_buffer(
                'table_tensor_' + str(i),
                torch.tensor(np.stack((np.real(item), np.imag(item))))
            )
        self.register_buffer('n_shift_tensor', torch.tensor(
            np.array(self.n_shift, dtype=np.double)))
        self.register_buffer('grid_size_tensor', torch.tensor(
            np.array(self.grid_size, dtype=np.double)))
        self.register_buffer('numpoints_tensor', torch.tensor(
            np.array(self.numpoints, dtype=np.double)))
        self.register_buffer('table_oversamp_tensor', torch.tensor(
            np.array(self.table_oversamp, dtype=np.double)))

    def forward(self, x, om, interp_mats=None):
        """Interpolate from gridded data to scattered data.

        Inputs are assumed to be batch/chans x coil x real/imag x image dims.
        Om should be nbatch x ndims x klength.

        Args:
            x (tensor): The DFT of the signal.
            om (tensor, optional): A new set of omega coordinates to
                interpolate to in radians/voxel.
            interp_mats (dict, default=None): A dictionary with keys
                'real_interp_mats' and 'imag_interp_mats', each key containing
                a list of interpolation matrices (see 
                mri.sparse_interp_mat.precomp_sparse_mats for construction).
                If None, then a standard interpolation is run.
        Returns:
            y (tensor): x computed at off-grid locations in om.
        """
        interpob = dict()
        interpob['table'] = []
        for i in range(len(self.table)):
            interpob['table'].append(getattr(self, 'table_tensor_' + str(i)))
        interpob['n_shift'] = self.n_shift_tensor
        interpob['grid_size'] = self.grid_size_tensor
        interpob['numpoints'] = self.numpoints_tensor
        interpob['table_oversamp'] = self.table_oversamp_tensor
        interpob['coil_broadcast'] = self.coil_broadcast
        interpob['matadj'] = self.matadj

        y = KbInterpFunction.apply(x, om, interpob, interp_mats)

        return y

    def __repr__(self):
        tablecheck = False
        out = '\nKbInterp forward interpolation object\n'
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            if 'table' in attr:
                if not tablecheck:
                    out = out + '   table: {} arrays, lengths: {}\n'.format(
                        len(self.table), self.table_oversamp)
                    tablecheck = True
            elif 'traj' in attr:
                out = out + '   traj: {} {} array\n'.format(
                    self.traj.shape, self.traj.dtype)
            elif 'interpob' not in attr:
                out = out + '   {}: {}\n'.format(attr, value)
        return out


class KbInterpBack(nn.Module):
    """Non-uniform FFT adjoint interpolation PyTorch layer.

    This object interpolates off-grid Fourier data to on-grid locations
    using a Kaiser-Bessel kernel. It is implemented as a PyTorch nn layer.

    Args:
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate to.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in each
            direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points to
            shift for fftshifts.
        table_oversamp (int, default=2^10): Table oversampling factor.
        order (ind, default=0): Order of Kaiser-Bessel kernel. Not currently
            implemented.
        coil_broadcast (boolean, default=False): Whether to broadcast across
            coil dimension. Much faster for many coils, but uses more memory.
        matadj (boolean, default=False): If true, adjoint interpolation
            constructs a sparse matrix and does the interpolation with the
            PyTorch sparse matrix API. (fastest option, more memory)
    """

    def __init__(self, im_size, grid_size=None, numpoints=6, n_shift=None,
                 table_oversamp=2**10, order=0, coil_broadcast=False,
                 matadj=False):

        super(KbInterpBack, self).__init__()

        self.alpha = 2.34
        self.im_size = im_size
        if grid_size is None:
            self.grid_size = tuple(np.array(self.im_size) * 2)
        else:
            self.grid_size = grid_size
        if n_shift is None:
            self.n_shift = tuple(np.array(self.im_size) // 2)
        else:
            self.n_shift = n_shift
        if numpoints == 6:
            self.numpoints = (6,) * len(self.grid_size)
        elif len(numpoints) != len(self.grid_size):
            self.numpoints = (numpoints,) * len(self.grid_size)
        else:
            self.numpoints = numpoints
        self.order = (0,)
        self.alpha = (2.34 * self.numpoints[0],)
        for i in range(1, len(self.numpoints)):
            self.alpha = self.alpha + (2.34 * self.numpoints[i],)
            self.order = self.order + (0,)
        if table_oversamp == 2**10:
            self.table_oversamp = (table_oversamp,) * len(self.grid_size)
        else:
            self.table_oversamp = table_oversamp
        table = build_table(
            numpoints=self.numpoints,
            table_oversamp=self.table_oversamp,
            grid_size=self.grid_size,
            im_size=self.im_size,
            ndims=len(self.im_size),
            order=self.order,
            alpha=self.alpha
        )
        self.table = table
        self.coil_broadcast = coil_broadcast
        self.matadj = matadj

        # dimension checking
        assert len(self.grid_size) == len(self.im_size)
        assert len(self.n_shift) == len(self.im_size)
        assert len(self.numpoints) == len(self.im_size)
        assert len(self.table) == len(self.im_size)
        assert len(self.table_oversamp) == len(self.im_size)
        assert len(self.alpha) == len(self.im_size)
        assert len(self.order) == len(self.im_size)

        self.table_tensor = []
        for i, item in enumerate(self.table):
            self.register_buffer(
                'table_tensor_' + str(i),
                torch.tensor(np.stack((np.real(item), np.imag(item))))
            )
            self.table_tensor.append(getattr(self, 'table_tensor_' + str(i)))
        self.register_buffer('n_shift_tensor', torch.tensor(
            np.array(self.n_shift, dtype=np.double)))
        self.register_buffer('grid_size_tensor', torch.tensor(
            np.array(self.grid_size, dtype=np.double)))
        self.register_buffer('numpoints_tensor', torch.tensor(
            np.array(self.numpoints, dtype=np.double)))
        self.register_buffer('table_oversamp_tensor', torch.tensor(
            np.array(self.table_oversamp, dtype=np.double)))

    def forward(self, y, om, interp_mats=None):
        """Interpolate from scattered data to gridded data.

        Apply table interpolation adjoint. Inputs are assumed to be batch/chans
        x coil x real/imag x kspace length. Om should be nbatch x ndims x
        klength.

        Args:
            y (tensor): The off-grid k-space data.
            om (tensor, optional): A new set of omega coordinates to
                interpolate from in radians/voxel.
            interp_mats (dict, default=None): A dictionary with keys
                'real_interp_mats' and 'imag_interp_mats', each key containing
                a list of interpolation matrices (see 
                mri.sparse_interp_mat.precomp_sparse_mats for construction).
                If None, then a standard interpolation is run.
        Returns:
            x (tensor): The DFT of the signal.
        """
        interpob = dict()
        interpob['table'] = []
        for i in range(len(self.table)):
            interpob['table'].append(getattr(self, 'table_tensor_' + str(i)))
        interpob['n_shift'] = self.n_shift_tensor
        interpob['grid_size'] = self.grid_size_tensor
        interpob['numpoints'] = self.numpoints_tensor
        interpob['table_oversamp'] = self.table_oversamp_tensor
        interpob['coil_broadcast'] = self.coil_broadcast
        interpob['matadj'] = self.matadj

        x = AdjKbInterpFunction.apply(y, om, interpob, interp_mats)

        return x

    def __repr__(self):
        tablecheck = False
        out = '\nKbInterp backward interpolation object\n'
        out = out + '----------------------------------------\n'
        for attr, value in self.__dict__.items():
            if 'table' in attr:
                if not tablecheck:
                    out = out + '   table: {} arrays, lengths: {}\n'.format(
                        len(self.table), self.table_oversamp)
                    tablecheck = True
            elif 'traj' in attr:
                out = out + '   traj: {} {} array\n'.format(
                    self.traj.shape, self.traj.dtype)
            elif 'interpob' not in attr:
                out = out + '   {}: {}\n'.format(attr, value)
        return out
