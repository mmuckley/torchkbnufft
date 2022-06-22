from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .utils import build_numpy_spmatrix, validate_args


def calc_tensor_spmatrix(
    omega: Tensor,
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2**10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""Builds a sparse matrix for interpolation.

    This builds the interpolation matrices directly from scipy Kaiser-Bessel
    functions, so using them for a NUFFT should be a little more accurate than
    table interpolation.

    This function has optional parameters for initializing a NUFFT object. See
    :py:class:`~torchkbnufft.KbNufft` for details.

    * :attr:`omega` should be of size ``(len(im_size), klength)``,
      where ``klength`` is the length of the k-space trajectory.

    Args:
        omega: k-space trajectory (in radians/voxel).
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
        device: Which device to create tensors on. Default:
            ``torch.device('cpu')``

    Returns:
        2-Tuple of (real, imaginary) tensors for NUFFT interpolation.

    Examples:

        >>> data = torch.randn(1, 1, 12) + 1j * torch.randn(1, 1, 12)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> spmats = tkbn.calc_tensor_spmatrix(omega, (8, 8))
        >>> adjkb_ob = tkbn.KbNufftAdjoint(im_size=(8, 8))
        >>> image = adjkb_ob(data, omega, spmats)
    """
    if not omega.ndim == 2:
        raise ValueError("Sparse matrix calculation not implemented for batched omega.")
    (
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        order,
        alpha,
        dtype,
        device,
    ) = validate_args(
        im_size,
        grid_size,
        numpoints,
        n_shift,
        table_oversamp,
        kbwidth,
        order,
        omega.dtype,
        omega.device,
    )
    coo = build_numpy_spmatrix(
        omega=omega.cpu().numpy(),
        numpoints=numpoints,
        im_size=im_size,
        grid_size=grid_size,
        n_shift=n_shift,
        order=order,
        alpha=alpha,
    )

    values = coo.data
    indices = np.stack((coo.row, coo.col))

    inds = torch.tensor(indices, dtype=torch.long, device=device)
    real_vals = torch.tensor(np.real(values), dtype=dtype, device=device)
    imag_vals = torch.tensor(np.imag(values), dtype=dtype, device=device)
    shape = coo.shape

    interp_mats = (
        torch.sparse.FloatTensor(inds, real_vals, torch.Size(shape)),  # type: ignore
        torch.sparse.FloatTensor(inds, imag_vals, torch.Size(shape)),  # type: ignore
    )

    return interp_mats
