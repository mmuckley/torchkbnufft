import warnings

import numpy as np
import torch

from .functional.kbnufft import (AdjKbNufftFunction, KbNufftFunction,
                                 ToepNufftFunction)
from .kbmodule import KbModule
from .nufft.utils import build_spmatrix, build_table, compute_scaling_coefs


class KbNufftModule(KbModule):
    """Parent class for KbNufft classes.

    This implementation collects all init functions into one place. It inherits
    from torch.nn.Module via torchkbnufft.kbmodule.KbModule.

    Args:
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate from.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in each
            direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points to
            shift for fftshifts.
        table_oversamp (int, default=2^10): Table oversampling factor.
        kbwidth (double, default=2.34): Kaiser-Bessel width parameter.
        order (double, default=0): Order of Kaiser-Bessel kernel.
        norm (str, default='None'): Normalization for FFT. Default uses no
            normalization. Use 'ortho' to use orthogonal FFTs and preserve
            energy.
    """

    def __init__(self, im_size, grid_size=None, numpoints=6, n_shift=None,
                 table_oversamp=2**10, kbwidth=2.34, order=0, norm='None',
                 coil_broadcast=False, matadj=False):
        super(KbNufftModule, self).__init__()

        self.im_size = im_size
        if grid_size is None:
            self.grid_size = tuple(np.array(self.im_size) * 2)
        else:
            self.grid_size = grid_size
        if n_shift is None:
            self.n_shift = tuple(np.array(self.im_size) // 2)
        else:
            self.n_shift = n_shift
        if isinstance(numpoints, int):
            self.numpoints = (numpoints,) * len(self.grid_size)
        else:
            self.numpoints = numpoints
        self.alpha = tuple(np.array(kbwidth) * np.array(self.numpoints))
        if isinstance(order, int) or isinstance(order, float):
            self.order = (order,) * len(self.grid_size)
        else:
            self.order = order
        if isinstance(table_oversamp, float) or isinstance(table_oversamp, int):
            self.table_oversamp = (table_oversamp,) * len(self.grid_size)
        else:
            self.table_oversamp = table_oversamp

        # dimension checking
        assert len(self.grid_size) == len(self.im_size)
        assert len(self.n_shift) == len(self.im_size)
        assert len(self.numpoints) == len(self.im_size)
        assert len(self.alpha) == len(self.im_size)
        assert len(self.order) == len(self.im_size)
        assert len(self.table_oversamp) == len(self.im_size)

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
        assert len(self.table) == len(self.im_size)

        scaling_coef = compute_scaling_coefs(
            im_size=self.im_size,
            grid_size=self.grid_size,
            numpoints=self.numpoints,
            alpha=self.alpha,
            order=self.order
        )
        self.scaling_coef = scaling_coef
        self.norm = norm
        self.coil_broadcast = coil_broadcast
        self.matadj = matadj

        if coil_broadcast == True:
            warnings.warn(
                'coil_broadcast will be deprecated in a future release',
                DeprecationWarning)
        if matadj == True:
            warnings.warn(
                'matadj will be deprecated in a future release',
                DeprecationWarning)

        self.register_buffer(
            'scaling_coef_tensor',
            torch.stack(
                (
                    torch.tensor(np.real(self.scaling_coef)),
                    torch.tensor(np.imag(self.scaling_coef))
                )
            )
        )
        for i, item in enumerate(self.table):
            self.register_buffer(
                'table_tensor_' + str(i),
                torch.tensor(np.stack((np.real(item), np.imag(item))))
            )
        self.register_buffer('n_shift_tensor', torch.tensor(
            np.array(self.n_shift, dtype=np.double)))
        self.register_buffer('grid_size_tensor', torch.tensor(
            np.array(self.grid_size, dtype=np.double)))
        self.register_buffer('im_size_tensor', torch.tensor(
            np.array(self.im_size, dtype=np.double)))
        self.register_buffer('numpoints_tensor', torch.tensor(
            np.array(self.numpoints, dtype=np.double)))
        self.register_buffer('table_oversamp_tensor', torch.tensor(
            np.array(self.table_oversamp, dtype=np.double)))

    def _extract_nufft_interpob(self):
        """Extracts interpolation object from self.

        Returns:
            dict: An interpolation object for the NUFFT operation.
        """
        interpob = dict()
        interpob['scaling_coef'] = self.scaling_coef_tensor
        interpob['table'] = []
        for i in range(len(self.table)):
            interpob['table'].append(getattr(self, 'table_tensor_' + str(i)))
        interpob['n_shift'] = self.n_shift_tensor
        interpob['grid_size'] = self.grid_size_tensor
        interpob['im_size'] = self.im_size_tensor
        interpob['numpoints'] = self.numpoints_tensor
        interpob['table_oversamp'] = self.table_oversamp_tensor
        interpob['norm'] = self.norm
        interpob['coil_broadcast'] = self.coil_broadcast
        interpob['matadj'] = self.matadj

        return interpob


class KbNufft(KbNufftModule):
    """Non-uniform FFT forward PyTorch module.

    This object applies the FFT and interpolates a grid of Fourier data to
    off-grid locations using a Kaiser-Bessel kernel. It is implemented as a
    PyTorch module.

    Args:
        im_size (int or tuple of ints): Size of base image.
        grid_size (int or tuple of ints, default=2*im_size): Size of the grid
            to interpolate from.
        numpoints (int or tuple of ints, default=6): Number of points to use
            for interpolation in each dimension. Default is six points in each
            direction.
        n_shift (int or tuple of ints, default=im_size//2): Number of points to
            shift for fftshifts.
        table_oversamp (int, default=2^10): Table oversampling factor.
        kbwidth (double, default=2.34): Kaiser-Bessel width parameter.
        order (double, default=0): Order of Kaiser-Bessel kernel.
        norm (str, default='None'): Normalization for FFT. Default uses no
            normalization. Use 'ortho' to use orthogonal FFTs and preserve
            energy.
    """

    def __init__(self, *args, **kwargs):
        super(KbNufft, self).__init__(*args, **kwargs)

    def forward(self, x, om, interp_mats=None):
        """Apply FFT and interpolate from gridded data to scattered data.

        Inputs are assumed to be batch/chans x coil x real/imag x image dims.
        Om should be nbatch x ndims x klength.

        Args:
            x (tensor): The original imagel.
            om (tensor, optional): A new set of omega coordinates at which to
                calculate the signal in radians/voxel.
            interp_mats (dict, default=None): A dictionary with keys
                'real_interp_mats' and 'imag_interp_mats', each key containing
                a list of interpolation matrices (see 
                mri.sparse_interp_mat.precomp_sparse_mats for construction).
                If None, then a standard interpolation is run.

        Returns:
            tensor: x computed at off-grid locations in om.
        """
        interpob = self._extract_nufft_interpob()

        y = KbNufftFunction.apply(x, om, interpob, interp_mats)

        return y


class AdjKbNufft(KbNufftModule):
    """Non-uniform FFT adjoint PyTorch module.

    This object interpolates off-grid Fourier data to on-grid locations
    using a Kaiser-Bessel kernel prior to inverse DFT. It is implemented as a
    PyTorch module.

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
        kbwidth (double, default=2.34): Kaiser-Bessel width parameter.
        order (double, default=0): Order of Kaiser-Bessel kernel.
        norm (str, default='None'): Normalization for FFT. Default uses no
            normalization. Use 'ortho' to use orthogonal FFTs and preserve
            energy.
    """

    def __init__(self, *args, **kwargs):
        super(AdjKbNufft, self).__init__(*args, **kwargs)

    def forward(self, y, om, interp_mats=None):
        """Interpolate from scattered data to gridded data and then iFFT.

        Inputs are assumed to be batch/chans x coil x real/imag x kspace
        length. Om should be nbatch x ndims x klength.

        Args:
            y (tensor): The off-grid signal.
            om (tensor, optional): The off-grid coordinates in radians/voxel.
            interp_mats (dict, default=None): A dictionary with keys
                'real_interp_mats' and 'imag_interp_mats', each key containing
                a list of interpolation matrices (see 
                mri.sparse_interp_mat.precomp_sparse_mats for construction).
                If None, then a standard interpolation is run.

        Returns:
            tensor: The image after adjoint NUFFT.
        """
        interpob = self._extract_nufft_interpob()

        x = AdjKbNufftFunction.apply(y, om, interpob, interp_mats)

        return x


class ToepNufft(KbModule):
    """Forward/backward NUFFT with Toeplitz embedding.

    This module applies Tx, where T is a matrix such that T = A'A, where A is
    a NUFFT matrix. Using Toeplitz embedding, this module computes the A'A
    operation without interpolations, which is extremely fast.

    The module is intended to be used in combination with an fft kernel
    computed to be the frequency response of an embedded Toeplitz matrix. The
    kernel is calculated offline via

    torchkbnufft.nufft.toep_functions.calc_toep_kernel

    The corresponding kernel is then passed to this module in its forward
    forward operation, which applies a (zero-padded) fft filter using the
    kernel.
    """

    def __init__(self):
        super(ToepNufft, self).__init__()

    def forward(self, x, kern, norm=None):
        """Toeplitz NUFFT forward function.

        Args:
            x (tensor): The image (or images) to apply the forward/backward
                Toeplitz-embedded NUFFT to.
            kern (tensor): The filter response taking into account Toeplitz
                embedding.
            norm (str, default=None): Use 'ortho' if kern was designed to use
                orthogonal FFTs.

        Returns:
            tensor: x after applying the Toeplitz NUFFT.
        """
        x = ToepNufftFunction.apply(x, kern, norm)

        return x
