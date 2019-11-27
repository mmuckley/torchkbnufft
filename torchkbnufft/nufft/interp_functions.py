import itertools

import numpy as np
import torch

from ..math import complex_mult, conj_complex_mult, imag_exp


def run_mat_interp(griddat, coef_mat_real, coef_mat_imag):
    """Interpolates griddat to off-grid coordinates with input sparse matrices.

    Args:
        griddat (tensor): The gridded frequency data.
        coef_mat_real (sparse tensor): The real interpolation coefficients
            stored as a sparse tensor.
        coef_mat_imag (sparse tensor): The imaginary interpolation coefficients
            stored as a sparse tensor.
        kdat (tensor): A tensor to store the outputs in.

    Returns:
        tensor: griddat interpolated to off-grid locations.
    """
    dtype = griddat.dtype
    device = griddat.device
    kdat_length = coef_mat_real.shape[0]

    # initialize output data
    kdat = torch.zeros(size=((griddat.shape[0], 2, kdat_length)),
                       dtype=dtype, device=device)

    real_griddat = griddat[:, 0, :].t()
    imag_griddat = griddat[:, 1, :].t()

    kdat[:, 0, :] = torch.mm(
        coef_mat_real,
        real_griddat
    ).t() - torch.mm(
        coef_mat_imag,
        imag_griddat
    ).t()

    kdat[:, 1, :] = torch.mm(
        coef_mat_real,
        imag_griddat
    ).t() + torch.mm(
        coef_mat_imag,
        real_griddat
    ).t()

    return kdat


def run_mat_interp_back(kdat, coef_mat_real, coef_mat_imag):
    """Interpolates kdat to on-grid coordinates with input sparse matrices.

    Args:
        kdat (tensor): The off-grid frequency data.
        coef_mat_real (sparse tensor): The real interpolation coefficients
            stored as a sparse tensor.
        coef_mat_imag (sparse tensor): The imaginary interpolation coefficients
            stored as a sparse tensor.
        griddat (tensor): A tensor to store the outputs in.

    Returns:
        tensor: kdat interpolated to on-grid locations.
    """
    dtype = kdat.dtype
    device = kdat.device
    griddat_length = coef_mat_real.shape[1]

    # initialize output data
    griddat = torch.zeros(size=((kdat.shape[0], 2, griddat_length)),
                          dtype=dtype, device=device)

    real_kdat = kdat[:, 0, :].t().reshape(-1, kdat.shape[0])
    imag_kdat = kdat[:, 1, :].t().reshape(-1, kdat.shape[0])
    coef_mat_real_t = coef_mat_real.t()
    coef_mat_imag_t = coef_mat_imag.t()

    # apply multiplies with complex conjugate
    griddat[:, 0, :] = torch.mm(
        coef_mat_real_t,
        real_kdat
    ).t() + torch.mm(
        coef_mat_imag_t,
        imag_kdat
    ).t()
    griddat[:, 1, :] = torch.mm(
        coef_mat_real_t,
        imag_kdat
    ).t() - torch.mm(
        coef_mat_imag_t,
        real_kdat
    ).t()

    return griddat


def calc_coef_and_indices(tm, kofflist, Jval, table, centers, L, dims, conjcoef=False):
    """Calculates interpolation coefficients and on-grid indices.

    Args:
        tm (tensor): normalized frequency locations.
        kofflist (tensor): A tensor with offset locations to first elements in
            list of nearest neighbords.
        Jval (tensor): A tuple-like tensor for how much to increment offsets.
        table (list): A list of tensors tabulating a Kaiser-Bessel
            interpolation kernel.
        centers (tensor): A tensor with the center locations of the table for
            each dimension.
        L (tensor): A tensor with the table size in each dimension.
        dims (tensor): A tensor with image dimensions.
        conjcoef (boolean, default=False): A boolean for whether to compute
            normal or complex conjugate interpolation coefficients
            (conjugate needed for adjoint).

    Returns:
        tuple: A tuple with interpolation coefficients and indices.
    """
    # type values
    dtype = tm.dtype
    device = tm.device
    int_type = torch.long

    # array shapes
    M = tm.shape[1]
    ndims = tm.shape[0]

    # indexing locations
    gridind = (kofflist + Jval.unsqueeze(1)).to(dtype)
    distind = torch.round(
        (tm - gridind) * L.unsqueeze(1)).to(dtype=int_type)
    gridind = gridind.to(int_type)

    arr_ind = torch.zeros((M,), dtype=int_type, device=device)
    coef = torch.stack((
        torch.ones(M, dtype=dtype, device=device),
        torch.zeros(M, dtype=dtype, device=device)
    ))

    for d in range(ndims):  # spatial dimension
        if conjcoef:
            coef = conj_complex_mult(
                coef,
                table[d][:, distind[d, :] + centers[d]],
                dim=0
            )
        else:
            coef = complex_mult(
                coef,
                table[d][:, distind[d, :] + centers[d]],
                dim=0
            )
        arr_ind = arr_ind + torch.remainder(gridind[d, :], dims[d]).view(-1) * \
            torch.prod(dims[d + 1:])

    return coef, arr_ind


def run_interp(griddat, tm, params):
    """Interpolates griddat to off-grid coordinates with input sparse matrices.

    Args:
        griddat (tensor): The on-grid frequency data.
        tm (tensor): Normalized frequency coordinates.
        params (dict): Dictionary with elements 'dims', 'table', 'numpoints',
            'Jlist', and 'table_oversamp'.

    Returns:
        tensor: griddat interpolated to off-grid locations.
    """
    # extract parameters
    dims = params['dims']
    table = params['table']
    numpoints = params['numpoints']
    Jlist = params['Jlist']
    L = params['table_oversamp']

    dtype = table[0].dtype
    device = table[0].device
    int_type = torch.long

    # center of tables
    centers = torch.floor(numpoints * L / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        torch.floor(tm - numpoints.unsqueeze(1) / 2.0).to(dtype=int_type)

    # initialize output array
    kdat = torch.zeros(size=((griddat.shape[0], 2, tm.shape[-1])),
                       dtype=dtype, device=device)

    # loop over offsets and take advantage of broadcasting
    for Jind in range(Jlist.shape[1]):
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, Jlist[:, Jind], table, centers, L, dims)

        # no danger of collisions for forward op
        arr_ind = arr_ind.unsqueeze(0).unsqueeze(0).expand(
            kdat.shape[0],
            kdat.shape[1],
            -1
        )
        kdat = kdat + complex_mult(
            coef.unsqueeze(0),
            torch.gather(griddat, 2, arr_ind),
            dim=1
        )

    return kdat


def run_interp_back(kdat, tm, params):
    """Interpolates kdat to on-grid coordinates.

    Args:
        kdat (tensor): The off-grid frequency data.
        tm (tensor): Normalized frequency coordinates.
        params (dict): Dictionary with elements 'dims', 'table', 'numpoints',
            'Jlist', and 'table_oversamp'.

    Returns:
        tensor: kdat interpolated to on-grid locations.
    """
    # extract parameters
    dims = params['dims']
    table = params['table']
    numpoints = params['numpoints']
    Jlist = params['Jlist']
    L = params['table_oversamp']

    # store data types
    dtype = table[0].dtype
    device = table[0].device
    int_type = torch.long

    # center of tables
    centers = torch.floor(numpoints * L / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    kofflist = 1 + \
        torch.floor(tm - numpoints.unsqueeze(1) / 2.0).to(dtype=torch.long)

    # initialize output array
    griddat = torch.zeros(size=(kdat.shape[0], 2, torch.prod(dims)),
                          dtype=dtype, device=device)

    # loop over offsets and take advantage of numpy broadcasting
    for Jind in range(Jlist.shape[1]):
        coef, arr_ind = calc_coef_and_indices(
            tm, kofflist, Jlist[:, Jind], table, centers, L, dims, conjcoef=True)

        if device == torch.device('cpu'):
            tmp = complex_mult(coef.unsqueeze(0), kdat, dim=1)
            for bind in range(griddat.shape[0]):
                for riind in range(griddat.shape[1]):
                    griddat[bind, riind].index_put_(
                        tuple(arr_ind.unsqueeze(0)),
                        tmp[bind, riind],
                        accumulate=True
                    )
        else:
            griddat.index_add_(
                2,
                arr_ind,
                complex_mult(coef.unsqueeze(0), kdat, dim=1)
            )

    return griddat


def kbinterp(x, om, interpob, interp_mats=None):
    """Apply table interpolation.

    Inputs are assumed to be batch/chans x coil x real/imag x image dims.
    Om should be nbatch x ndims x klength.

    Args:
        x (tensor): The oversampled DFT of the signal.
        om (tensor, optional): A custom set of k-space points to
            interpolate to in radians/voxel.
        interpob (dict): An interpolation object with 'table', 'n_shift',
            'grid_size', 'numpoints', and 'table_oversamp' keys. See
            models.kbinterp.py for details.
        interp_mats (dict, default=None): A dictionary with keys
            'real_interp_mats' and 'imag_interp_mats', each key containing a
            list of interpolation matrices (see
            mri.sparse_interp_mat.precomp_sparse_mats for construction). If
            None, then a standard interpolation is run.

    Returns:
        tensor: The signal interpolated to off-grid locations.
    """
    dtype = interpob['table'][0].dtype
    device = interpob['table'][0].device

    # extract interpolation params
    n_shift = interpob['n_shift']
    grid_size = interpob['grid_size']
    numpoints = interpob['numpoints']

    ndims = om.shape[1]

    # convert to normalized freq locs
    tm = torch.zeros(size=om.shape, dtype=dtype, device=device)
    Jgen = []
    for i in range(ndims):
        gam = (2 * np.pi / grid_size[i])
        tm[:, i, :] = om[:, i, :] / gam
        Jgen.append(range(np.array(numpoints[i].cpu(), dtype=np.int)))

    # build an iterator for going over all J values
    Jgen = list(itertools.product(*Jgen))
    Jgen = torch.tensor(Jgen).permute(1, 0).to(dtype=torch.long, device=device)

    # set up params if not using sparse mats
    if interp_mats is None:
        params = {
            'dims': None,
            'table': interpob['table'],
            'numpoints': numpoints,
            'Jlist': Jgen,
            'table_oversamp': interpob['table_oversamp'],
        }

    y = []
    # run the table interpolator for each batch element
    for b in range(x.shape[0]):
        if interp_mats is None:
            params['dims'] = torch.tensor(
                x[b].shape[2:], dtype=torch.long, device=device)

            y.append(run_interp(x[b].view((x.shape[1], 2, -1)), tm[b], params))
        else:
            y.append(
                run_mat_interp(
                    x[b].view((x.shape[1], 2, -1)),
                    interp_mats['real_interp_mats'][b],
                    interp_mats['imag_interp_mats'][b],
                )
            )

        # phase for fftshift
        y[-1] = complex_mult(
            y[-1],
            imag_exp(torch.mv(torch.transpose(
                om[b], 1, 0), n_shift)).unsqueeze(0),
            dim=1
        )

    y = torch.stack(y)

    return y


def adjkbinterp(y, om, interpob, interp_mats=None):
    """Apply table interpolation adjoint.

    Inputs are assumed to be batch/chans x coil x real/imag x kspace length.
    Om should be nbatch x ndims x klength.

    Args:
        y (tensor): The off-grid DFT of the signal.
        om (tensor, optional): A set of k-space points to
            interpolate from in radians/voxel.
        interpob (dict): An interpolation object with 'table', 'n_shift',
            'grid_size', 'numpoints', and 'table_oversamp' keys. See
            models.kbinterp.py for details.
        interp_mats (dict, default=None): A dictionary with keys
            'real_interp_mats' and 'imag_interp_mats', each key containing a
            list of interpolation matrices (see
            mri.sparse_interp_mat.precomp_sparse_mats for construction). If
            None, then a standard interpolation is run.

    Returns:
        tensor: The signal interpolated to on-grid locations.
    """
    y = y.clone()

    n_shift = interpob['n_shift']
    grid_size = interpob['grid_size']
    numpoints = interpob['numpoints']

    dtype = interpob['table'][0].dtype
    device = interpob['table'][0].device

    ndims = om.shape[1]

    # convert to normalized freq locs
    tm = torch.zeros(size=om.shape, dtype=dtype, device=device)
    Jgen = []
    for i in range(ndims):
        gam = 2 * np.pi / grid_size[i]
        tm[:, i, :] = om[:, i, :] / gam
        Jgen.append(range(np.array(numpoints[i].cpu(), np.int)))

    # build an iterator for going over all J values
    Jgen = list(itertools.product(*Jgen))
    Jgen = torch.tensor(Jgen).permute(1, 0).to(dtype=torch.long, device=device)

    # set up params if not using sparse mats
    if interp_mats is None:
        params = {
            'dims': None,
            'table': interpob['table'],
            'numpoints': numpoints,
            'Jlist': Jgen,
            'table_oversamp': interpob['table_oversamp'],
        }

    x = []
    # run the table interpolator for each batch element
    for b in range(y.shape[0]):
        # phase for fftshift
        y[b] = conj_complex_mult(
            y[b],
            imag_exp(torch.mv(torch.transpose(
                om[b], 1, 0), n_shift)).unsqueeze(0),
            dim=1
        )

        if interp_mats is None:
            params['dims'] = grid_size.to(dtype=torch.long, device=device)

            x.append(run_interp_back(y[b], tm[b], params))
        else:
            x.append(
                run_mat_interp_back(
                    y[b],
                    interp_mats['real_interp_mats'][b],
                    interp_mats['imag_interp_mats'][b],
                )
            )

    x = torch.stack(x)

    bsize = y.shape[0]
    ncoil = y.shape[1]
    out_size = (bsize, ncoil, 2) + tuple(grid_size.to(torch.long))

    x = x.view(out_size)

    return x
