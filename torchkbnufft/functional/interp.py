from typing import List, Tuple

import torch
from torch import Tensor

from .._autograd.interp import (
    KbSpmatInterpAdjoint,
    KbSpmatInterpForward,
    KbTableInterpAdjoint,
    KbTableInterpForward,
)


def kb_spmat_interp(image: Tensor, interp_mats: Tuple[Tensor, Tensor]) -> Tensor:
    """Kaiser-Bessel sparse matrix interpolation.

    See :py:class:`~torchkbnufft.KbInterp` for an overall description of
    interpolation.

    To calculate the sparse matrix tuple, see
    :py:meth:`~torchkbnufft.calc_tensor_spmatrix`.

    Args:
        image: Gridded data to be interpolated to scattered data.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.

    Returns:
        ``image`` calculated at scattered locations.
    """
    is_complex = True
    if not image.is_complex():
        if not image.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        image = torch.view_as_complex(image)

    data = KbSpmatInterpForward.apply(image, interp_mats)

    if is_complex is False:
        data = torch.view_as_real(data)

    return data


def kb_spmat_interp_adjoint(
    data: Tensor, interp_mats: Tuple[Tensor, Tensor], grid_size: Tensor
) -> Tensor:
    """Kaiser-Bessel sparse matrix interpolation adjoint.

    See :py:class:`~torchkbnufft.KbInterpAdjoint` for an overall description of
    adjoint interpolation.

    To calculate the sparse matrix tuple, see
    :py:meth:`~torchkbnufft.calc_tensor_spmatrix`.

    Args:
        data: Scattered data to be interpolated to gridded data.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.

    Returns:
        ``data`` calculated at gridded locations.
    """
    is_complex = True
    if not data.is_complex():
        if not data.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        data = torch.view_as_complex(data)

    image = KbSpmatInterpAdjoint.apply(data, interp_mats, grid_size)

    if is_complex is False:
        image = torch.view_as_real(image)

    return image


def kb_table_interp(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Kaiser-Bessel table interpolation.

    See :py:class:`~torchkbnufft.KbInterp` for an overall description of
    interpolation and how to construct the function arguments.

    Args:
        image: Gridded data to be interpolated to scattered data.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift: Size for fftshift, usually ``im_size // 2``.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            ``numpoints``.

    Returns:
        ``image`` calculated at scattered locations.
    """
    is_complex = True
    if not image.is_complex():
        if not image.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        image = torch.view_as_complex(image)

    data = KbTableInterpForward.apply(
        image, omega, tables, n_shift, numpoints, table_oversamp, offsets
    )

    if is_complex is False:
        data = torch.view_as_real(data)

    return data


def kb_table_interp_adjoint(
    data: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    grid_size: Tensor,
) -> Tensor:
    """Kaiser-Bessel table interpolation adjoint.

    See :py:class:`~torchkbnufft.KbInterpAdjoint` for an overall description of
    adjoint interpolation.

    Args:
        data: Scattered data to be interpolated to gridded data.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift: Size for fftshift, usually ``im_size // 2``.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            ``numpoints``.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``.

    Returns:
        ``data`` calculated at gridded locations.
    """
    is_complex = True
    if not data.is_complex():
        if not data.shape[-1] == 2:
            raise ValueError("For real inputs, last dimension must be size 2.")

        is_complex = False
        data = torch.view_as_complex(data)

    image = KbTableInterpAdjoint.apply(
        data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
    )

    if is_complex is False:
        image = torch.view_as_real(image)

    return image
