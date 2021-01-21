from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

import torchkbnufft as tkbn
import torchkbnufft.functional as tkbnF

from .._nufft.utils import compute_scaling_coefs
from .kbmodule import KbModule


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

    def __init__(
        self,
        im_size: Sequence[int],
        grid_size: Optional[Sequence[int]] = None,
        numpoints: Union[int, Sequence[int]] = 6,
        n_shift: Optional[Sequence[int]] = None,
        table_oversamp: Union[int, Sequence[int]] = 2 ** 10,
        kbwidth: float = 2.34,
        order: Union[float, Sequence[float]] = 0.0,
        dtype: torch.dtype = None,
    ):
        super().__init__(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints,
            n_shift=n_shift,
            table_oversamp=table_oversamp,
            kbwidth=kbwidth,
            order=order,
            dtype=dtype,
        )

        scaling_coef = compute_scaling_coefs(
            im_size=self.im_size.tolist(),  # type: ignore
            grid_size=self.grid_size.tolist(),  # type: ignore
            numpoints=self.numpoints.tolist(),  # type: ignore
            alpha=self.alpha.tolist(),  # type: ignore
            order=self.order.tolist(),  # type: ignore
        )

        self.register_buffer(
            "scaling_coef", scaling_coef.to(self.table_0.dtype)  # type: ignore
        )


class KbNufft(KbNufftModule):
    """Non-uniform FFT forward PyTorch module.

    This object applies the FFT and interpolates a grid of Fourier data to
    off-grid locations using a Kaiser-Bessel kernel. It is implemented as a
    PyTorch module.
    """

    def forward(
        self,
        image: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        smaps: Optional[Tensor] = None,
        norm: Optional[str] = None,
    ) -> Tensor:
        """Apply FFT and interpolate from gridded data to scattered data.

        Inputs are assumed to be batch/chans x coil x real/imag x image dims.
        Om should be nbatch x ndims x klength.
        """
        if smaps is not None:
            if not smaps.dtype == image.dtype:
                raise TypeError("image dtype does not match smaps dtype.")

        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError("For real inputs, last dimension must be size 2.")

                smaps = torch.view_as_complex(smaps)

            is_complex = False
            image = torch.view_as_complex(image)

        if smaps is not None:
            image = image * smaps

        if interp_mats is not None:
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)

            output = tkbnF.kb_spmat_nufft(
                image=image,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                interp_mats=interp_mats,
                norm=norm,
            )
        else:
            tables = []
            for i in range(len(self.im_size)):  # type: ignore
                tables.append(getattr(self, f"table_{i}"))

            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)

            output = tkbnF.kb_table_nufft(
                image=image,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                omega=omega,
                tables=tables,
                n_shift=self.n_shift,
                numpoints=self.numpoints,
                table_oversamp=self.table_oversamp,
                offsets=self.offsets.to(torch.long),
                norm=norm,
            )

        if not is_complex:
            output = torch.view_as_real(output)

        return output


class KbNufftAdjoint(KbNufftModule):
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

    def forward(
        self,
        data: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        smaps: Optional[Tensor] = None,
        norm: Optional[str] = None,
    ) -> Tensor:
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
        if smaps is not None:
            if not smaps.dtype == data.dtype:
                raise TypeError("data dtype does not match smaps dtype.")

        is_complex = True
        if not data.is_complex():
            if not data.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError("For real inputs, last dimension must be size 2.")

                smaps = torch.view_as_complex(smaps)

            is_complex = False
            data = torch.view_as_complex(data)

        if interp_mats is not None:
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)

            output = tkbnF.kb_spmat_nufft_adjoint(
                data=data,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                interp_mats=interp_mats,
                norm=norm,
            )
        else:
            tables = []
            for i in range(len(self.im_size)):  # type: ignore
                tables.append(getattr(self, f"table_{i}"))

            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)

            output = tkbnF.kb_table_nufft_adjoint(
                data=data,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                omega=omega,
                tables=tables,
                n_shift=self.n_shift,
                numpoints=self.numpoints,
                table_oversamp=self.table_oversamp,
                offsets=self.offsets.to(torch.long),
                norm=norm,
            )

        if smaps is not None:
            output = torch.sum(output * smaps.conj(), dim=1, keepdim=True)

        if not is_complex:
            output = torch.view_as_real(output)

        return output


class ToepNufft(torch.nn.Module):
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
        super().__init__()

    def forward(
        self,
        image: Tensor,
        kernel: Tensor,
        smaps: Optional[Tensor] = None,
        norm: Optional[str] = None,
    ) -> Tensor:
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
        if smaps is not None:
            if not smaps.dtype == image.dtype:
                raise TypeError("image dtype does not match smaps dtype.")

        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError("For real inputs, last dimension must be size 2.")

                smaps = torch.view_as_complex(smaps)

            is_complex = False
            image = torch.view_as_complex(image)

        if smaps is None:
            output = tkbnF.fft_filter(image=image, kernel=kernel, norm=norm)
        else:
            output = []

            # do a batch loop to conserve memory
            for (mini_image, smap) in zip(image, smaps):
                mini_image = mini_image.unsqueeze(0) * smap.unsqueeze(0)
                mini_image = tkbnF.fft_filter(
                    image=mini_image, kernel=kernel, norm=norm
                )
                mini_image = torch.sum(
                    mini_image * smap.unsqueeze(0).conj(),
                    dim=1,
                    keepdim=True,
                )
                output.append(mini_image.squeeze(0))

            output = torch.stack(output)

        if not is_complex:
            output = torch.view_as_real(output)

        return output
