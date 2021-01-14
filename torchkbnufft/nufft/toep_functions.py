import copy
import itertools

import numpy as np
import torch

from ..math import absolute


def calc_toep_kernel(adj_ob, om, weights=None):
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
    dtype, device = om.dtype, om.device
    adj_ob = copy.deepcopy(adj_ob).to(dtype=dtype, device=device)
    ndims = om.shape[1]

    # remove sensitivities if dealing with MriSenseNufft
    if "Sense" in adj_ob.__class__.__name__:
        adj_ob.smap_tensor = torch.ones(
            adj_ob.smap_tensor.shape, dtype=dtype, device=device
        )
        adj_ob.smap_tensor = adj_ob.smap_tensor[:, 0:1]
        adj_ob.smap_tensor[:, :, 1] = 0

    # remove this because we won't need it
    adj_ob.n_shift = tuple(np.array(adj_ob.n_shift) * 0)
    adj_ob.n_shift_tensor = adj_ob.n_shift_tensor * 0

    # if we don't have any weights, just use ones
    if weights is None:
        weights = torch.stack((torch.ones(om.shape[-1]), torch.zeros(om.shape[-1]))).to(
            dtype=dtype, device=device
        )
        weights = weights.unsqueeze(0).unsqueeze(0)
    elif weights.shape[2] == 1:
        weights = torch.cat(
            (weights, torch.zeros(weights.shape, dtype=dtype, device=device)), 2
        )

    flip_list = list(itertools.product(*list([range(2)] * (ndims - 1))))
    base_flip = torch.tensor([1], dtype=dtype, device=device)

    kern = []
    for ind in range(om.shape[0]):
        kern.append(
            _get_kern(om[ind].unsqueeze(0), weights, flip_list, base_flip, adj_ob)
        )

    kern = torch.cat(kern, dim=0)

    return kern


def _get_kern(om, weights, flip_list, base_flip, adj_ob):
    """Calculates a single FFT kernel for Toeplitz embedding.

    This function is called by calc_toep_kernel() in a loop.

    Args:
        om (tensor): The k-space trajectory in radians/voxel.
        weights (tensor, default=None): Non-Cartesian k-space weights (e.g.,
            density compensation).
        flip_list (list): A list of flipping directions.
        base_flip (tensor): A base flip list.
        adj_ob (object): The adjoint NUFFT object.

    Returns:
        tensor: The FFT kernel for approximating the forward/backward
            operation.
    """
    dtype, device = om.dtype, om.device
    ndims = om.shape[1]
    kern = []

    # flip across each dimension except last to get full kernel
    for flips in flip_list:
        flip_coef = torch.cat(
            (base_flip, torch.tensor(flips, dtype=dtype, device=device) * -2 + 1)
        )
        flip_coef = flip_coef.unsqueeze(0).unsqueeze(-1)

        tmp_om = om * flip_coef

        kern.append(adj_ob(weights, tmp_om))

        for dim, el in enumerate(flips):
            if el == 1:
                kern[-1] = kern[-1].flip(dim)

    # concatenate all calculated blocks, walking back from last dim
    for dim in range(ndims - 1):
        kern = cat_blocks(kern, dim)
    kern = kern[0]

    # now that we have half the kernel we can use Hermitian symmetry
    kern = reflect_conj_concat(kern, ndims - 1)

    # make sure kernel is Hermitian symmetric
    kern = hermitify(kern, ndims - 1)

    permute_dims = (0, 1) + tuple(range(3, kern.ndim)) + (2,)
    inv_permute_dims = (0, 1, kern.ndim - 1) + tuple(range(2, kern.ndim - 1))

    # put the kernel in fft space
    kern = torch.fft(kern.permute(permute_dims), kern.ndim - 3).permute(
        inv_permute_dims
    )

    if adj_ob.norm == "ortho":
        kern = kern / torch.sqrt(torch.prod(torch.tensor(kern.shape[3:], dtype=dtype)))

    return kern


def cat_blocks(blocks, dim):
    """Concatenates pairwise a list of blocks along dim.

    This function concatenates pairwise elements from blocks along dimension
    dim, removing the first element from the even blocks and replacing it with
    zeros. The removal is done to comply with Toeplitz embedding conventions.

    Args:
        blocks (list): A list of tensors to concatenate.
        dim (int): Dimension along which to concatenate.

    Returns:
        list: Another list of tensor blocks after pairwise concatenation.
    """
    dtype, device = blocks[0].dtype, blocks[0].device
    kern = []
    dim = -1 - dim

    # calculate the size of the zero block
    zblockshape = torch.tensor(blocks[0].shape)
    zblockshape[dim] = 1
    zblock = torch.zeros(*zblockshape, dtype=dtype, device=device)

    # loop over pairwise elements, concatenating with the zero block and
    # removing duplicates
    for ind in range(0, len(blocks), 2):
        tmpblock = blocks[ind + 1].narrow(dim, 1, blocks[ind + 1].shape[dim] - 1)
        tmpblock = torch.cat((zblock, tmpblock.flip(dim)), dim)
        kern.append(torch.cat((blocks[ind], tmpblock), dim))

    return kern


def reflect_conj_concat(kern, dim):
    """Reflects and conjugates kern before concatenating along dim.

    Args:
        kern (tensor): One half of a full, Hermitian-symmetric kernel.
        dim (int): The integer across which to apply Hermitian symmetry.

    Returns:
        tensor: The full FFT kernel after Hermitian-symmetric reflection.
    """
    dtype, device = kern.dtype, kern.device
    dim = -1 - dim
    flipdims = tuple(torch.arange(abs(dim)) + dim)

    # calculate size of central z block
    zblockshape = torch.tensor(kern.shape)
    zblockshape[dim] = 1
    zblock = torch.zeros(*zblockshape, dtype=dtype, device=device)

    # conjugation array
    conj_arr = torch.tensor([1, -1], dtype=dtype, device=device)
    conj_arr = conj_arr.unsqueeze(0).unsqueeze(0)
    while conj_arr.ndim < kern.ndim:
        conj_arr = conj_arr.unsqueeze(-1)

    # reflect the original block and conjugate it
    tmpblock = conj_arr * kern
    for d in flipdims:
        tmpblock = tmpblock.index_select(
            d,
            torch.remainder(
                -1 * torch.arange(tmpblock.shape[d], device=device), tmpblock.shape[d]
            ),
        )
    tmpblock = torch.cat(
        (zblock, tmpblock.narrow(dim, 1, tmpblock.shape[dim] - 1)), dim
    )

    # concatenate and return
    return torch.cat((kern, tmpblock), dim)


def hermitify(kern, dim):
    """Enforce Hermitian symmetry.

    This function takes an approximately Hermitian-symmetric kernel and
    enforces Hermitian symmetry by calcualting a tensor that reverses the
    coordinates and conjugates the original, then averaging that tensor with
    the original.

    Args:
        kern (tensor): An approximately Hermitian-symmetric kernel.
        dim (int): The last imaging dimension.

    Returns:
        tensor: A Hermitian-symmetric kernel.
    """
    dtype, device = kern.dtype, kern.device
    dim = -1 - dim + kern.ndim

    start = kern.clone()

    # reverse coordinates for each dimension
    for d in range(dim, kern.ndim):
        kern = kern.index_select(
            d,
            torch.remainder(
                -1 * torch.arange(kern.shape[d], device=device), kern.shape[d]
            ),
        )

    # conjugate
    conj_arr = torch.tensor([1, -1], dtype=dtype, device=device)
    conj_arr = conj_arr.unsqueeze(0).unsqueeze(0)
    while conj_arr.ndim < kern.ndim:
        conj_arr = conj_arr.unsqueeze(-1)
    kern = conj_arr * kern

    # take the average
    kern = (start + kern) / 2

    return kern
