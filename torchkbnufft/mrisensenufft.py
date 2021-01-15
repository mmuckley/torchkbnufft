import warnings

import numpy as np
import torch

from .functional.mrisensenufft import (
    AdjMriSenseNufftFunction,
    MriSenseNufftFunction,
    ToepSenseNufftFunction,
)
from .kbmodule import KbModule
from .mri.sensenufft_functions import sense_toeplitz
from .nufft.utils import build_spmatrix, build_table, compute_scaling_coefs


class SenseNufftModule(KbModule):
    """Parent class for SENSE-NUFFT classes.

    This implementation collects all init functions into one place. It inherits
    from torch.nn.Module via torchkbnufft.KbModule.

    Args:
        smap (tensor): Sensitivity coils of size (batch_size, real/imag,) +
            im_size.
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
        coilpack (boolean, default=False): If True, packs batch dimension into
            coil dimension prior to NUFFT. This is useful when batch is a set
            of slices with 1 k-space trajectory (note: coilpack expects ktraj
            to have batch dim of 1).
    """

    def __init__(
        self,
        smap,
        im_size,
        grid_size=None,
        numpoints=6,
        n_shift=None,
        table_oversamp=2 ** 10,
        kbwidth=2.34,
        order=0,
        norm="None",
        coil_broadcast=False,
        coilpack=False,
        matadj=False,
    ):
        super(SenseNufftModule, self).__init__()

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
        if isinstance(table_oversamp, int):
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
            alpha=self.alpha,
        )
        self.table = table
        assert len(self.table) == len(self.im_size)

        scaling_coef = compute_scaling_coefs(
            im_size=self.im_size,
            grid_size=self.grid_size,
            numpoints=self.numpoints,
            alpha=self.alpha,
            order=self.order,
        )
        self.scaling_coef = scaling_coef
        self.norm = norm
        self.coil_broadcast = coil_broadcast
        self.coilpack = coilpack
        self.matadj = matadj
        self.smap_shape = smap.shape

        if coil_broadcast == True:
            warnings.warn(
                "coil_broadcast will be deprecated in a future release",
                DeprecationWarning,
            )
        if matadj == True:
            warnings.warn(
                "matadj will be deprecated in a future release", DeprecationWarning
            )

        self.register_buffer(
            "scaling_coef_tensor",
            torch.stack(
                (
                    torch.tensor(np.copy(np.real(self.scaling_coef))),
                    torch.tensor(np.copy(np.imag(self.scaling_coef))),
                )
            ),
        )
        for i, item in enumerate(self.table):
            self.register_buffer(
                "table_tensor_" + str(i),
                torch.tensor(np.stack((np.real(item), np.imag(item)))),
            )
        self.register_buffer("smap_tensor", smap)
        self.register_buffer(
            "n_shift_tensor", torch.tensor(np.array(self.n_shift, dtype=np.double))
        )
        self.register_buffer(
            "grid_size_tensor", torch.tensor(np.array(self.grid_size, dtype=np.double))
        )
        self.register_buffer(
            "im_size_tensor", torch.tensor(np.array(self.im_size, dtype=np.double))
        )
        self.register_buffer(
            "numpoints_tensor", torch.tensor(np.array(self.numpoints, dtype=np.double))
        )
        self.register_buffer(
            "table_oversamp_tensor",
            torch.tensor(np.array(self.table_oversamp, dtype=np.double)),
        )

    def _extract_sense_interpob(self):
        """Extracts interpolation object from self.

        Returns:
            dict: An interpolation object for the NUFFT operation.
        """
        interpob = dict()
        interpob["scaling_coef"] = self.scaling_coef_tensor
        interpob["table"] = []
        for i in range(len(self.table)):
            interpob["table"].append(getattr(self, "table_tensor_" + str(i)))
        interpob["n_shift"] = self.n_shift_tensor
        interpob["grid_size"] = self.grid_size_tensor
        interpob["im_size"] = self.im_size_tensor
        interpob["numpoints"] = self.numpoints_tensor
        interpob["table_oversamp"] = self.table_oversamp_tensor
        interpob["norm"] = self.norm
        interpob["coil_broadcast"] = self.coil_broadcast
        interpob["coilpack"] = self.coilpack
        interpob["matadj"] = self.matadj

        return interpob


class MriSenseNufft(SenseNufftModule):
    """Non-uniform FFT forward PyTorch module with SENSE.

    This object multiplies sensitivity coils and then applies a Kaiser-Bessel
    kernel NUFFT. It is implemented as a PyTorch module.

    Args:
        smap (tensor): Sensitivity coils of size (batch_size, real/imag,) +
            im_size.
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
        coilpack (boolean, default=False): If True, packs batch dimension into
            coil dimension prior to NUFFT. This is useful when batch is a set
            of slices with 1 k-space trajectory (note: coilpack expects ktraj
            to have batch dim of 1).
    """

    def __init__(self, *args, **kwargs):
        super(MriSenseNufft, self).__init__(*args, **kwargs)

    def forward(self, x, om, interp_mats=None, smap=None):
        """Apply SENSE maps and NUFFT.

        Inputs are assumed to be batch/chans x coil (1) x real/imag x image dims.
        Om should be nbatch x ndims x klength (when not using coil packing).

        Args:
            x (tensor): The original image.
            om (tensor, optional): A new set of omega coordinates at which to
                calculate the signal.
            interp_mats (dict, default=None): A dictionary with keys
                'real_interp_mats' and 'imag_interp_mats', each key containing
                a list of interpolation matrices (see
                mri.sparse_interp_mat.precomp_sparse_mats for construction).
                If None, then a standard interpolation is run.
            smap (tensor, default=None): If passed in, uses input smap tensor
                instead of the one used on layer initialization.

        Returns:
            tensor: x computed at off-grid locations in om.
        """
        interpob = self._extract_sense_interpob()

        if smap is None:
            smap = self.smap_tensor

        y = MriSenseNufftFunction.apply(x, smap, om, interpob, interp_mats)

        return y


class AdjMriSenseNufft(SenseNufftModule):
    """Non-uniform FFT adjoint PyTorch module with SENSE.

    This object applies a NUFFT adjoint, then a SENSE adjoint.

    Args:
        smap (tensor): Sensitivity coils of size (batch_size, real/imag,) +
            im_size.
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
        coilpack (boolean, default=False): If True, packs batch dimension into
            coil dimension prior to NUFFT. This is useful when batch is a set
            of slices with 1 k-space trajectory (note: coilpack expects ktraj
            to have batch dim of 1).
    """

    def __init__(self, *args, **kwargs):
        super(AdjMriSenseNufft, self).__init__(*args, **kwargs)

    def forward(self, y, om, interp_mats=None, smap=None):
        """Apply adjoint NUFFT and SENSE.

        Inputs are assumed to be batch/chans x coil x real/imag x kspace length.
        Om should be nbatch x ndims x klength (when not using coil packing).

        Args:
            y (tensor): The off-grid signal.
            om (tensor, optional): The off-grid coordinates.
            interp_mats (dict, default=None): A dictionary with keys
                'real_interp_mats' and 'imag_interp_mats', each key containing
                a list of interpolation matrices (see
                mri.sparse_interp_mat.precomp_sparse_mats for construction).
                If None, then a standard interpolation is run.
            smap (tensor, default=None): If passed in, uses input smap tensor
                instead of the one used on layer initialization.

        Returns:
            tensor: The image with an adjoint SENSE-NUFFT.
        """
        interpob = self._extract_sense_interpob()

        if smap is None:
            smap = self.smap_tensor

        x = AdjMriSenseNufftFunction.apply(y, smap, om, interpob, interp_mats)

        return x


class ToepSenseNufft(KbModule):
    """Forward/backward SENSE-NUFFT with Toeplitz embedding.

    This module applies Tx, where T = A'A and A is a SENSE-NUFFT matrix. The
    module computes this operation without NUFFT interpolations, enabling very
    fast computation speeds. It accomplishes this with a precomputed FFT kernel
    that embeds the Toeplitz matrix into an FFT matrix. The FFT kernel can be
    precomputed using

    torchkbnufft.nufft.toep_functions.calc_toep_kernel

    The corresponding kernel is then passed to this module in its forward
    forward operation, which applies a SENSE expansion, a (zero-padded) fft
    filter using the kernel, and then finally a SENSE reduction.

    Args:
        smap (tensor): Sensitivity coils of size (batch_size, real/imag,) +
            im_size.
    """

    def __init__(self, smap):
        super(ToepSenseNufft, self).__init__()

        self.smap_shape = smap.shape

        self.register_buffer("smap_tensor", smap)

    def forward(self, x, kern, smap=None, norm=None):
        """Toeplitz SENSE-NUFFT forward function.

        Args:
            x (tensor): The image (or images) to apply the forward/backward
                Toeplitz-embedded NUFFT to.
            kern (tensor): The filter response taking into account Toeplitz
                embedding.
            smap (tensor, default=None): If passed in, uses input smap tensor
                instead of the one used on layer initialization.
            norm (str, default=None): Use 'ortho' if kern was designed to use
                orthogonal FFTs.

        Returns:
            tensor: x after applying the Toeplitz NUFFT.
        """
        if smap is None:
            smap = self.smap_tensor

        x = ToepSenseNufftFunction.apply(x, smap, kern, norm)

        return x
