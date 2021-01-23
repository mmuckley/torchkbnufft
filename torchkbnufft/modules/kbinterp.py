from typing import Optional, Sequence, Tuple, Union

import torch
import torchkbnufft.functional as tkbnF
from torch import Tensor

from ._kbmodule import KbModule


class KbInterpModule(KbModule):
    """Parent class for KbInterp classes.

    Not all args are necessary to specify. If you only provide ``im_size``, the
    code will try to infer reasonable defaults based on ``im_size``.

    Args:
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        numpoints: Number of neighbors to use for interpolation.
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers.
        device: Which device to create tensors on.
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
    """Non-uniform KB interpolation layer.

    This object interpolates a grid of Fourier data to off-grid locations
    using a Kaiser-Bessel kernel.

    Not all args are necessary to specify. If you only provide ``im_size``, the
    code will try to infer reasonable defaults based on ``im_size``.

    Args:
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        numpoints: Number of neighbors to use for interpolation.
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers.
        device: Which device to create tensors on.
    """

    def forward(
        self,
        image: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Interpolate from gridded data to scattered data.

        Input tensors should be of shape `(N, C) + im_size`, where `N` is the
        batch size and `C` is the number of sensitivity coils. `omega`, the
        k-space trajectory, should be of size `(len(im_size), klength)`, where
        `klength` is the length of the k-space trajectory.

        If your tensors are real, ensure that 2 is the size of the last
        dimension.

        Args:
            image: Gridded data to be interpolated to scattered data.
            omega: k-space trajectory (in radians/voxel).
            interp_mats; Optional: 2-tuple of real, imaginary sparse matrices
                to use for sparse matrix KB interpolation (overrides default
                table interpolation).

        Returns:
            `image` calculated at Fourier frequencies specified by `omega`.
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

    Not all args are necessary to specify. If you only provide ``im_size``, the
    code will try to infer reasonable defaults based on ``im_size``.

    Args:
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        numpoints: Number of neighbors to use for interpolation.
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers.
        device: Which device to create tensors on.
    """

    def forward(
        self,
        data: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        grid_size: Optional[Tensor] = None,
    ) -> Tensor:
        """Interpolate from scattered data to gridded data.

        Input tensors should be of shape `(N, C) + klength`, where `N` is the
        batch size and `C` is the number of sensitivity coils. `omega`, the
        k-space trajectory, should be of size `(len(im_size), klength)`, where
        `klength` is the length of the k-space trajectory.

        If your tensors are real, ensure that 2 is the size of the last
        dimension.

        Args:
            data: Data to be gridded and then inverse FFT'd.
            omega: k-space trajectory (in radians/voxel).
            interp_mats; Optional: 2-tuple of real, imaginary sparse matrices
                to use for sparse matrix KB interpolation (overrides default
                table interpolation).

        Returns:
            `data` transformed to the image domain.
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
