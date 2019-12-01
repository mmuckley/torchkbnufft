import itertools

import numpy as np
import torch

from ..math import complex_mult, conj_complex_mult
from .interp_functions import calc_coef_and_indices


def get_interpob(model):
    """Retrieves the interpolation dictionary from model.

    Different nufft objects use different interpolation objects. This function
    only extracts the minimum amount necessary for sparse matrix
    precomputation.

    Args:
        model (TorchKbNufft object): A TorchKbNufft object with attributes for
            forming a KbNufft interpolation dictionary.

    Returns:
        dict: A dictionary with interpolation parameters.
    """
    interpob = dict()
    interpob['table'] = []
    for i in range(len(model.table)):
        interpob['table'].append(getattr(model, 'table_tensor_' + str(i)))
    interpob['grid_size'] = model.grid_size_tensor
    interpob['numpoints'] = model.numpoints_tensor
    interpob['table_oversamp'] = model.table_oversamp_tensor

    return interpob


def compute_forw_mat(dims, table, numpoints, Jlist, L, tm):
    """Compute a forward Kaiser-Bessel interpolation sparse matrix.

    Args:
        dims (tensor): A list of sizes of each dimension.
        table (tensor): A list of interpolation tables.
        numpoints (tensor): A list of numbers of nearest neighbors for each
            dimension.
        Jlist (tensor): A list of nearest neighbor configurations.
        L (tensor): A list of table sizes for each dimension.
        tm (tensor): An array of normalized frequency locations.

    Returns:
        tuple: A 2-length tuple with a sparse interpolation matrix in each
            element. The first matrix has the real coefficients; the second
            has the imaginary.
    """
    dtype = table[0].dtype
    device = table[0].device
    int_type = torch.long

    nJ = Jlist.shape[1]

    # center of tables
    centers = torch.floor(numpoints * L / 2).to(dtype=int_type)
    # offset from k-space to first coef loc
    kofflist = 1 + torch.floor(tm - numpoints.unsqueeze(1) / 2.0)

    # do a bit of type management - ints for faster index comps
    dims = dims.to(dtype=int_type)
    kofflist = kofflist.to(dtype=int_type)
    Jlist = Jlist.to(dtype=int_type)

    # initialize the sparse matrices
    coef_mat_real = torch.sparse.FloatTensor(
        tm.shape[-1], torch.prod(dims)).to(dtype=dtype, device=device)
    coef_mat_imag = torch.sparse.FloatTensor(
        tm.shape[-1], torch.prod(dims)).to(dtype=dtype, device=device)

    # loop over offsets and take advantage of broadcasting
    for Jind in range(nJ):
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, Jlist[:, Jind], table, centers, L, dims)

        sparse_coords = torch.stack(
            (
                torch.arange(
                    arr_ind.shape[0],
                    dtype=arr_ind.dtype,
                    device=arr_ind.device
                ),
                arr_ind
            )
        )
        coef_mat_real = coef_mat_real + torch.sparse.FloatTensor(
            sparse_coords,
            coef[0],
            torch.Size((arr_ind.shape[0], torch.prod(dims)))
        )
        coef_mat_imag = coef_mat_imag + torch.sparse.FloatTensor(
            sparse_coords,
            coef[1],
            torch.Size((arr_ind.shape[0], torch.prod(dims)))
        )

    return coef_mat_real, coef_mat_imag


def precomp_sparse_mats(om, model):
    """Precompute sparse interpolation matrices.

    Args:
        om (tensor): The k-space trajectory in radians/voxel.
        model (TorchKbNufft object): A KbNufft type object with attributes for
            creating a KbNufft interpolation object.

    Returns:
        tuple: A 2-length tuple with lists of sparse interpolation matrices in
            each element. The first matrix has the real coefficient matrices;
            the second has the imaginary.
    """
    interpob = get_interpob(model)

    dtype = interpob['table'][0].dtype
    device = interpob['table'][0].device

    # extract interpolation params and match device and dtype to input
    table = interpob['table']
    grid_size = interpob['grid_size']
    numpoints = interpob['numpoints']
    table_oversamp = interpob['table_oversamp']

    ndims = om.shape[1]
    M = om.shape[2]

    # convert to normalized freq locs
    tm = torch.zeros(size=om.shape, dtype=dtype, device=device)
    Jgen = []
    for i in range(ndims):
        gam = (2 * np.pi / grid_size[i])
        tm[:, i, :] = om[:, i, :] / gam
        Jgen.append(range(np.array(numpoints[i].cpu(), dtype=np.int)))

    # build an iterator for going over all J values
    Jgen = list(itertools.product(*Jgen))

    coef_real_mats = []
    coef_imag_mats = []
    for norm_traj in tm:
        coef_mat_real, coef_mat_imag = compute_forw_mat(
            grid_size.to(dtype=dtype, device=device),
            table,
            numpoints,
            torch.tensor(
                np.transpose(np.array(Jgen)),
                dtype=dtype,
                device=device
            ),
            table_oversamp,
            norm_traj
        )

        coef_real_mats.append(coef_mat_real)
        coef_imag_mats.append(coef_mat_imag)

    return coef_real_mats, coef_imag_mats
