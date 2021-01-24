from typing import Optional, Sequence, Tuple, Union

import torch
import torchkbnufft.functional as tkbnF
from torch import Tensor

from ._kbmodule import KbModule


class KbInterpModule(KbModule):
    """Parent class for KbInterp classes.

    See subclasses for an explanation of inputs.
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
    r"""Non-uniform Kaiser-Bessel interpolation layer.

    This object interpolates a grid of Fourier data to off-grid locations
    using a Kaiser-Bessel kernel. Consider an off-grid signal :math:`Y` to be
    estimated from an on-grid signal :math:`X`, then in one dimension this
    layer applies:

    .. math::
        Y(\omega_m) = \sum_{j=1}^J X_{\{k_m+j\}_K}u^*_j(\omega_m)

    where :math:`u` is the Kaiser-Bessel kernel, :math:`\omega` is the k-space
    coordinates in radians/voxel, :math:`k_m` is the index of the nearest
    sample of :math:`X` to frequency location :math:`\omega_m`, and :math:`J`
    is the number of nearest neighbors to use from the Kaiser-Bessel kernel.
    For a detailed description see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://ieeexplore.ieee.org/abstract/document/1166689>`_.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image (used for calculating the
      kernel but not for the actual operation).

    * :attr:`grid_size` is the size of the grid after forward FFT. To reduce
      errors, NUFFT operations are done on an oversampled grid to reduce
      interpolation distances. This will typically be 1.25 to 2 times
      :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors of the kernel to use
      for interpolation.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for fftshift. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on.
            Default: ``torch.device('cpu')``
    """

    def forward(
        self,
        image: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Interpolate from gridded data to scattered data.

        Input tensors should be of shape ``(N, C) + im_size``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size ``(len(im_size), klength)``,
        where ``klength`` is the length of the k-space trajectory.

        If your tensors are real-valued, ensure that 2 is the size of the last
        dimension.

        Args:
            image: Gridded data to be interpolated to scattered data.
            omega: k-space trajectory (in radians/voxel).
            interp_mats: 2-tuple of real, imaginary sparse matrices to use for
                sparse matrix KB interpolation (overrides default table
                interpolation).

        Returns:
            ``image`` calculated at Fourier frequencies specified by ``omega``.
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
    r"""Non-uniform FFT adjoint interpolation PyTorch layer.

    This object interpolates off-grid Fourier data to on-grid locations using a
    Kaiser-Bessel kernel. Consider an off-grid signal :math:`Y` to be
    interpolated to an on-grid signal :math:`X`, then in one dimension this
    layer applies:

    .. math::
        X_{\{k_m+j\}_K} = \sum_{j=1}^J Y(\omega_m) u_j(\omega_m)

    where :math:`u` is the Kaiser-Bessel kernel, :math:`\omega` is the k-space
    coordinates in radians/voxel, :math:`k_m` is the index of the nearest
    sample of :math:`X` to frequency location :math:`\omega_m`, and :math:`J`
    is the number of nearest neighbors to use from the Kaiser-Bessel kernel.
    For a detailed description see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://ieeexplore.ieee.org/abstract/document/1166689>`_.

    Note:

        This function is not the inverse of ``KbInterp``, is is the adjoint.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image (used for calculating the
      kernel but not for the actual operation).

    * :attr:`grid_size` is the size of the grid after adjoint interpolation. To
      reduce errors, NUFFT operations are done on an oversampled grid to reduce
      interpolation distances. This will typically be 1.25 to 2 times
      :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors of the kernel to use
      for interpolation.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension.
        n_shift: Size for fftshift. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.
        dtype: Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device: Which device to create tensors on.
            Default: ``torch.device('cpu')``
    """

    def forward(
        self,
        data: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        grid_size: Optional[Tensor] = None,
    ) -> Tensor:
        """Interpolate from scattered data to gridded data.

        Input tensors should be of shape ``(N, C) + klength``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size ``(len(im_size), klength)``,
        where ``klength`` is the length of the k-space trajectory.

        If your tensors are real-valued, ensure that 2 is the size of the last
        dimension.

        Args:
            data: Data to be gridded.
            omega: k-space trajectory (in radians/voxel).
            interp_mats: 2-tuple of real, imaginary sparse matrices to use for
                sparse matrix KB interpolation (overrides default table
                interpolation).

        Returns:
            ``data`` transformed to the image domain.
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
