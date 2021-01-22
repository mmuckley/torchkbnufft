from typing import Optional, Sequence, Tuple, Union

import torch
import torchkbnufft.functional as tkbnF
from torch import Tensor

from ._kbmodule import KbModule


class KbInterpModule(KbModule):
    """Parent class for KbInterp classes.

    This implementation collects all init functions into one place. It inherits
    from torch.nn.Module via torchkbnufft.kbmodule.KbModule.

    Args:
        im_size: Size of base image.
        grid_size; Optional: Size of the grid to interpolate to.
        numpoints: Number of points to use for interpolation in each dimension.
            Default is six points in each direction.
        n_shift; Optional: Number of points to shift for fftshifts.
        table_oversamp: Table oversampling factor.
        kbwidth: Kaiser-Bessel width parameter.
        order: Order of Kaiser-Bessel kernel.
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
        device: torch.device = None,
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
            device=device,
        )


class KbInterp(KbInterpModule):
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
        kbwidth (double, default=2.34): Kaiser-Bessel width parameter.
        order (double, default=0): Order of Kaiser-Bessel kernel.
    """

    def forward(
        self,
        image: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Interpolate from gridded data to scattered data.

        Inputs are assumed to be batch/chans x coil x real/imag x image dims.
        Om should be nbatch x ndims x klength.

        Args:
            image: The DFT of the signal.
            omega: A new set of omega coordinates to interpolate to in
                radians/voxel.
            interp_mats: Tuple with two interpolation matrices (real, imag).

        Returns:
            image computed at off-grid locations in omega.
        """
        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")

            is_complex = False
            image = torch.view_as_complex(image)

        if interp_mats is not None:
            output = tkbnF.kb_spmat_interp(image=image, interp_mats=interp_mats)
        else:
            tables = []
            for i in range(len(self.im_size)):  # type: ignore
                tables.append(getattr(self, f"table_{i}"))

            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)

            output = tkbnF.kb_table_interp(
                image=image,
                omega=omega,
                tables=tables,
                n_shift=self.n_shift,
                numpoints=self.numpoints,
                table_oversamp=self.table_oversamp,
                offsets=self.offsets.to(torch.long),
            )

        if not is_complex:
            output = torch.view_as_real(output)

        return output


class KbInterpAdjoint(KbInterpModule):
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
        kbwidth (double, default=2.34): Kaiser-Bessel width parameter.
        order (double, default=0): Order of Kaiser-Bessel kernel.
    """

    def forward(
        self,
        data: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        grid_size: Optional[Tensor] = None,
    ) -> Tensor:
        """Interpolate from scattered data to gridded data.

        Apply table interpolation adjoint. Inputs are assumed to be batch/chans
        x coil x real/imag x kspace length. Om should be nbatch x ndims x
        klength.

        Args:
            data: The off-grid k-space data.
            omega: The coordinates of the off-grid dtaa.
            interp_mats: Tuple with two interpolation matrices (real, imag).

        Returns:
            data computed at on-grid locations.
        """
        is_complex = True
        if not data.is_complex():
            if not data.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")

            is_complex = False
            data = torch.view_as_complex(data)

        if grid_size is None:
            assert isinstance(self.grid_size, Tensor)
            grid_size = self.grid_size
        if interp_mats is not None:
            output = tkbnF.kb_spmat_interp_adjoint(
                data=data, interp_mats=interp_mats, grid_size=grid_size
            )
        else:
            tables = []
            for i in range(len(self.im_size)):  # type: ignore
                tables.append(getattr(self, f"table_{i}"))

            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)

            output = tkbnF.kb_table_interp_adjoint(
                data=data,
                omega=omega,
                tables=tables,
                n_shift=self.n_shift,
                numpoints=self.numpoints,
                table_oversamp=self.table_oversamp,
                offsets=self.offsets,
                grid_size=grid_size,
            )

        if not is_complex:
            output = torch.view_as_real(output)

        return output
