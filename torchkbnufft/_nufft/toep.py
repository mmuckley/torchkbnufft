from typing import Optional, Sequence, Union

import torch
import torchkbnufft as tkbn
from torch import Tensor

from ..modules import KbNufftAdjoint
from .fft import fft_fn


def calculate_toeplitz_kernel(
    omega: Tensor,
    im_size: Sequence[int],
    weights: Optional[Tensor] = None,
    norm: Optional[str] = None,
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2 ** 10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
) -> Tensor:
    """Calculates an FFT kernel for Toeplitz embedding over batches.

    The kernel is calculated using a adjoint NUFFT object. If the adjoint
    applies A', then this script calculates D where F'DF = A'WA, where F is a
    DFT matrix and W is a set of non-Cartesian k-space weights. D can then be
    used to approximate A'WA without any interpolation operations.

    Args:
        adj_ob (object): The adjoint NUFFT object.
        om (tensor): The k-space trajectory in radians/voxel.
        weights (tensor, default=None): Non-Cartesian k-space weights (e.g.,
            density compensation). Optional.

    Returns:
        tensor: The FFT kernel for approximating the forward/backward
            operation for all batches.
    """
    device = omega.device
    normalized = True if norm == "ortho" else False

    adj_ob = tkbn.KbNufftAdjoint(
        im_size=im_size,
        grid_size=grid_size,
        numpoints=numpoints,
        n_shift=[0 for _ in range(omega.shape[0])],
        table_oversamp=table_oversamp,
        kbwidth=kbwidth,
        order=order,
        dtype=omega.dtype,
        device=omega.device,
    )

    # if we don't have any weights, just use ones
    assert isinstance(adj_ob.table_0, Tensor)
    if weights is None:
        weights = torch.ones(omega.shape[-1], dtype=adj_ob.table_0.dtype, device=device)
        weights = weights.unsqueeze(0).unsqueeze(0)
    else:
        weights = weights.to(adj_ob.table_0)

    # apply adjoints to n-1 dimensions
    if omega.shape[0] > 1:
        kernel = adjoint_flip_and_concat(1, omega, weights, adj_ob, norm)
    else:
        kernel = adj_ob(weights, omega, norm=norm)

    # now that we have half the kernel
    # we can use Hermitian symmetry
    kernel = reflect_conj_concat(kernel, 2)

    # make sure kernel is Hermitian symmetric
    kernel = hermitify(kernel, 2)

    # put the kernel in fft space
    return fft_fn(kernel, omega.shape[0], normalized=normalized)


def adjoint_flip_and_concat(
    dim: int,
    omega: Tensor,
    weights: Tensor,
    adj_ob: KbNufftAdjoint,
    norm: Optional[str] = None,
) -> Tensor:
    im_dim = dim + 2

    if dim < omega.shape[0] - 1:
        kernel1 = adjoint_flip_and_concat(dim + 1, omega, weights, adj_ob, norm)
        flip_coef = torch.ones(
            omega.shape[0], dtype=omega.dtype, device=omega.device
        ).unsqueeze(-1)
        flip_coef[dim] = -1
        kernel2 = adjoint_flip_and_concat(
            dim + 1, flip_coef * omega, weights, adj_ob, norm
        )
    else:
        kernel1 = adj_ob(weights, omega, norm=norm)
        flip_coef = torch.ones(
            omega.shape[0], dtype=omega.dtype, device=omega.device
        ).unsqueeze(-1)
        flip_coef[dim] = -1
        kernel2 = adj_ob(weights, flip_coef * omega, norm=norm)

    # calculate the size of the zero block
    zero_block_shape = torch.tensor(kernel1.shape)
    zero_block_shape[im_dim] = 1
    zero_block = torch.zeros(
        *zero_block_shape, dtype=kernel1.dtype, device=kernel1.device
    )

    # remove zero freq and concat
    kernel2 = kernel2.narrow(im_dim, 1, kernel2.shape[im_dim] - 1)

    return torch.cat((kernel1, zero_block, kernel2.flip(im_dim)), im_dim)


def reflect_conj_concat(kernel: Tensor, dim: int) -> Tensor:
    """Reflects and conjugates kern before concatenating along dim.

    Args:
        kern (tensor): One half of a full, Hermitian-symmetric kernel.
        dim (int): The integer across which to apply Hermitian symmetry.

    Returns:
        tensor: The full FFT kernel after Hermitian-symmetric reflection.
    """
    dtype, device = kernel.dtype, kernel.device
    flipdims = torch.arange(dim, kernel.ndim, device=device)

    # calculate size of central z block
    zero_block_shape = torch.tensor(kernel.shape, device=device)
    zero_block_shape[dim] = 1
    zero_block = torch.zeros(*zero_block_shape, dtype=dtype, device=device)

    # reflect the original block and conjugate it
    # the below code looks a bit hacky but we don't want to flip the 0 dim
    # TODO: make this better
    tmp_block = kernel.conj()
    for d in flipdims:
        tmp_block = tmp_block.index_select(
            d,
            torch.remainder(
                -1 * torch.arange(tmp_block.shape[d], device=device), tmp_block.shape[d]
            ),
        )
    tmp_block = torch.cat(
        (zero_block, tmp_block.narrow(dim, 1, tmp_block.shape[dim] - 1)), dim
    )

    # concatenate and return
    return torch.cat((kernel, tmp_block), dim)


def hermitify(kernel: Tensor, dim: int) -> Tensor:
    """Enforce Hermitian symmetry.

    This function takes an approximately Hermitian-symmetric kernel and
    enforces Hermitian symmetry by calcualting a tensor that reverses the
    coordinates and conjugates the original, then averaging that tensor with
    the original.

    Args:
        kernel: An approximately Hermitian-symmetric kernel.
        dim: The last imaging dimension.

    Returns:
        A Hermitian-symmetric kernel.
    """
    device = kernel.device

    start = kernel.clone()

    # reverse coordinates for each dimension
    # the below code looks a bit hacky but we don't want to flip the 0 dim
    # TODO: make this better
    for d in range(dim, kernel.ndim):
        kernel = kernel.index_select(
            d,
            torch.remainder(
                -1 * torch.arange(kernel.shape[d], device=device), kernel.shape[d]
            ),
        )

    return (start + kernel.conj()) / 2
