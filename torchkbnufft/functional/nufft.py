from typing import List, Optional, Tuple

import torch
from torch import Tensor

from torchkbnufft._nufft.fft import fft_and_scale, ifft_and_scale
from torchkbnufft.functional.interp import (
    kb_spmat_interp,
    kb_spmat_interp_adjoint,
    kb_table_interp,
    kb_table_interp_adjoint,
)


def kb_spmat_nufft(
    image: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    interp_mats: Tuple[Tensor, Tensor],
    norm: Optional[str] = None,
) -> Tensor:
    """Kaiser-Bessel NUFFT with sparse matrix interpolation.

    See :py:class:`~torchkbnufft.KbNufft` for an overall description of
    the forward NUFFT.

    To calculate the sparse matrix tuple, see
    :py:meth:`~torchkbnufft.calc_tensor_spmatrix`.

    Args:
        image: Image to be NUFFT'd to scattered data.
        scaling_coef: Image-domain coefficients to pre-compensate for
            interpolation errors.
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.
        norm: Whether to apply normalization with the FFT operation. Options
            are ``"ortho"`` or ``None``.

    Returns:
        ``image`` calculated at scattered Fourier locations.
    """
    is_complex = True
    if not image.is_complex():
        if not image.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        image = torch.view_as_complex(image)

    data = kb_spmat_interp(
        image=fft_and_scale(
            image=image,
            scaling_coef=scaling_coef,
            im_size=im_size,
            grid_size=grid_size,
            norm=norm,
        ),
        interp_mats=interp_mats,
    )

    if is_complex is False:
        data = torch.view_as_real(data)

    return data


def kb_spmat_nufft_adjoint(
    data: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    interp_mats: Tuple[Tensor, Tensor],
    norm: Optional[str] = None,
) -> Tensor:
    """Kaiser-Bessel adjoint NUFFT with sparse matrix interpolation.

    See :py:class:`~torchkbnufft.KbNufftAdjoint` for an overall description of
    the adjoint NUFFT.

    To calculate the sparse matrix tuple, see
    :py:meth:`~torchkbnufft.calc_tensor_spmatrix`.

    Args:
        data: Scattered data to be iNUFFT'd to an image.
        scaling_coef: Image-domain coefficients to compensate for interpolation
            errors.
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.
        norm: Whether to apply normalization with the FFT operation. Options
            are ``"ortho"`` or ``None``.

    Returns:
        ``data`` transformed to an image.
    """
    is_complex = True
    if not data.is_complex():
        if not data.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        data = torch.view_as_complex(data)

    image = ifft_and_scale(
        image=kb_spmat_interp_adjoint(
            data=data, interp_mats=interp_mats, grid_size=grid_size
        ),
        scaling_coef=scaling_coef,
        im_size=im_size,
        grid_size=grid_size,
        norm=norm,
    )

    if is_complex is False:
        image = torch.view_as_real(image)

    return image


def kb_table_nufft(
    image: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    """Kaiser-Bessel NUFFT with table interpolation.

    See :py:class:`~torchkbnufft.KbNufft` for an overall description of
    the forward NUFFT.

    Args:
        image: Image to be NUFFT'd to scattered data.
        scaling_coef: Image-domain coefficients to pre-compensate for
            interpolation errors.
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift: Size for fftshift, usually ``im_size // 2``.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            `numpoints`.
        norm: Whether to apply normalization with the FFT operation.
            Options are ``"ortho"`` or ``None``.

    Returns:
        ``image`` calculated at scattered Fourier locations.
    """
    is_complex = True
    if not image.is_complex():
        if not image.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        image = torch.view_as_complex(image)

    data = kb_table_interp(
        image=fft_and_scale(
            image=image,
            scaling_coef=scaling_coef,
            im_size=im_size,
            grid_size=grid_size,
            norm=norm,
        ),
        omega=omega,
        tables=tables,
        n_shift=n_shift,
        numpoints=numpoints,
        table_oversamp=table_oversamp,
        offsets=offsets,
    )

    if is_complex is False:
        data = torch.view_as_real(data)

    return data


def kb_table_nufft_adjoint(
    data: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    """Kaiser-Bessel NUFFT adjoint with table interpolation.

    See :py:class:`~torchkbnufft.KbNufftAdjoint` for an overall description of
    the adjoint NUFFT.

    Args:
        data: Scattered data to be iNUFFT'd to an image.
        scaling_coef: Image-domain coefficients to compensate for interpolation
            errors.
        im_size: Size of image with length being the number of dimensions.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift: Size for fftshift, usually ``im_size // 2``.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            `numpoints`.
        norm: Whether to apply normalization with the FFT operation.
            Options are ``"ortho"`` or ``None``.

    Returns:
        ``data`` transformed to an image.
    """
    is_complex = True
    if not data.is_complex():
        if not data.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        data = torch.view_as_complex(data)

    image = ifft_and_scale(
        image=kb_table_interp_adjoint(
            data=data,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
            grid_size=grid_size,
        ),
        scaling_coef=scaling_coef,
        im_size=im_size,
        grid_size=grid_size,
        norm=norm,
    )

    if is_complex is False:
        image = torch.view_as_real(image)

    return image
