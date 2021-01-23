from typing import List, Tuple

from torch import Tensor

from .._autograd.interp import (
    KbSpmatInterpAdjoint,
    KbSpmatInterpForward,
    KbTableInterpAdjoint,
    KbTableInterpForward,
)


def kb_spmat_interp(image: Tensor, interp_mats: Tuple[Tensor, Tensor]) -> Tensor:
    """Kaiser-Bessel sparse matrix interpolation.

    Args:
        image: Gridded data to be interpolated to scattered data.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.

    Returns:
        `image` calculated at scattered locations.
    """
    return KbSpmatInterpForward.apply(image, interp_mats)


def kb_spmat_interp_adjoint(
    data: Tensor, interp_mats: Tuple[Tensor, Tensor], grid_size: Tensor
) -> Tensor:
    """Kaiser-Bessel sparse matrix interpolation adjoint.

    Args:
        data: Scattered data to be interpolated to gridded data.
        interp_mats: 2-tuple of real, imaginary sparse matrices to use for
            sparse matrix KB interpolation.

    Returns:
        `data` calculated at gridded locations.
    """
    return KbSpmatInterpAdjoint.apply(data, interp_mats, grid_size)


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

    Args:
        image: Gridded data to be interpolated to scattered data.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            `numpoints`.

    Returns:
        `image` calculated at scattered locations.
    """
    return KbTableInterpForward.apply(
        image, omega, tables, n_shift, numpoints, table_oversamp, offsets
    )


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

    Args:
        data: Scattered data to be interpolated to gridded data.
        omega: k-space trajectory (in radians/voxel).
        tables: Interpolation tables (one table for each dimension).
        n_shift; Optional: Size for fftshift, usually `im_size // 2`.
        numpoints: Number of neighbors to use for interpolation.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            `numpoints`.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.

    Returns:
        `data` calculated at gridded locations.
    """
    return KbTableInterpAdjoint.apply(
        data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
    )
