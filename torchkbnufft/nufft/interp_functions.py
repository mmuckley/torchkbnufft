import itertools

import numpy as np
import torch

from ..math import complex_mult, conj_complex_mult, imag_exp


def run_mat_interp(griddat, coef_mat_real, coef_mat_imag, kdat):
    """Interpolates griddat to off-grid coordinates with input sparse matrices.

    Args:
        griddat (tensor): The gridded frequency data.
        coef_mat_real (sparse tensor): The real interpolation coefficients stored
            as a sparse tensor.
        coef_mat_imag (sparse tensor): The imaginary interpolation coefficients stored
            as a sparse tensor.
        kdat (tensor): A tensor to store the outputs in.
    Returns:
        kdat (tensor): griddat interpolated to off-grid locations.
    """
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


def run_mat_interp_back(kdat, coef_mat_real, coef_mat_imag, griddat):
    """Interpolates kdat to on-grid coordinates with input sparse matrices.

    Args:
        kdat (tensor): The off-grid frequency data.
        coef_mat_real (sparse tensor): The real interpolation coefficients stored
            as a sparse tensor.
        coef_mat_imag (sparse tensor): The imaginary interpolation coefficients stored
            as a sparse tensor.
        griddat (tensor): A tensor to store the outputs in.
    Returns:
        griddat (tensor): kdat interpolated to on-grid locations.
    """
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


def run_interp(griddat, dims, table, numpoints, Jlist, L, tm, kdat):
    """Interpolates griddat to off-grid coordinates with input sparse matrices.

    Args:
        griddat (tensor): The on-grid frequency data.
        dims (tensor): A list of size of each dimension.
        table (list): A list of interpolation tables (one table for each dimension).
        numpoints (tensor): A list of number of nearest neighbors to use for each
            dimension.
        Jlist (tensor): An array with all possible combinations of offsets.
        L (tensor): Number of samples in table.
        tm (tensor): Normalized frequency coordinates.
        kdat (tensor): A tensor to store the outputs in.
    Returns:
        kdat (tensor): griddat interpolated to off-grid locations.
    """
    dtype = table[0].dtype
    device = table[0].device
    int_type = torch.long

    M = tm.shape[1]
    ndims = tm.shape[0]
    nJ = Jlist.shape[1]

    # center of tables
    centers = torch.floor(numpoints * L / 2).to(dtype=int_type)
    # offset from k-space to first coef loc
    kofflist = 1 + torch.floor(tm - numpoints.unsqueeze(1) / 2.0)

    # do a bit of type management - ints for faster index comps
    curgridind = torch.zeros(tm.shape, dtype=dtype, device=device)
    curdistind = torch.zeros(tm.shape, dtype=int_type, device=device)
    arr_ind = torch.zeros((M,), dtype=int_type, device=device)
    coef = torch.ones((2, M), dtype=dtype, device=device)
    dims = dims.to(dtype=int_type)
    kofflist = kofflist.to(dtype=int_type)
    Jlist = Jlist.to(dtype=int_type)

    # loop over offsets and take advantage of broadcasting
    for Jind in range(nJ):
        curgridind = (kofflist + Jlist[:, Jind].unsqueeze(1)).to(dtype)
        curdistind = torch.round(
            (tm - curgridind) * L.unsqueeze(1)).to(dtype=int_type)
        curgridind = curgridind.to(int_type)

        arr_ind = torch.zeros((M,), dtype=int_type, device=device)
        coef = torch.stack((
            torch.ones(M, dtype=dtype, device=device),
            torch.zeros(M, dtype=dtype, device=device)
        ))

        for d in range(ndims):  # spatial dimension
            coef = complex_mult(
                coef,
                table[d][:, curdistind[d, :] + centers[d]],
                dim=0
            )
            arr_ind = arr_ind + torch.remainder(curgridind[d, :], dims[d]).view(-1) * \
                torch.prod(dims[d + 1:])

        # no danger of collisions for forward op
        kdat = kdat + complex_mult(
            coef.unsqueeze(0),
            griddat[:, :, arr_ind],
            dim=1
        )

    return kdat


def run_interp_back(kdat, dims, table, numpoints, Jlist, L, tm, griddat,
                    coil_broadcast=False, matadj=False):
    """Interpolates kdat to on-grid coordinates.

    Args:
        kdat (tensor): The off-grid frequency data.
        dims (tensor): A list of size of each dimension.
        table (list): A list of interpolation tables (one table for each dimension).
        numpoints (tensor): A list of number of nearest neighbors to use for each
            dimension.
        Jlist (tensor): An array with all possible combinations of offsets.
        L (int): Number of samples in table.
        tm (tensor): Normalized frequency coordinates.
        griddat (tensor): A tensor to store the outputs in.
        coil_broadcast (boolean, default=False): If True, tries to do broacast
            multiplies across coil dimension. Can be slower or faster, but always
            uses more memory.
        matadj (boolean, default=False): If True, this function will construct
            a sparse interpolation matrix and use this matrix to do the
            interpolation. This is the fastest option, but requires more memory.
    Returns:
        griddat (tensor): kdat interpolated to on-grid locations.
    """
    dtype = table[0].dtype
    device = table[0].device
    int_type = torch.long

    M = tm.shape[1]
    ndims = tm.shape[0]
    nJ = Jlist.shape[1]

    # center of tables
    centers = torch.floor(numpoints * L / 2).to(dtype=int_type)
    # offset from k-space to first coef loc
    kofflist = 1 + torch.floor(tm - numpoints.unsqueeze(1) / 2.0)

    # do a bit of type management - ints for faster index comps
    curgridind = torch.zeros(tm.shape, dtype=dtype, device=device)
    curdistind = torch.zeros(tm.shape, dtype=int_type, device=device)
    arr_ind = torch.zeros((M,), dtype=int_type, device=device)
    coef = torch.ones((2, M), dtype=dtype, device=device)
    dims = dims.to(dtype=int_type)
    kofflist = kofflist.to(dtype=int_type, device=device)
    Jlist = Jlist.to(dtype=int_type)

    coef_mat_real = torch.sparse.FloatTensor(torch.prod(
        dims), kdat.shape[-1]).to(dtype=dtype, device=device)
    coef_mat_imag = torch.sparse.FloatTensor(torch.prod(
        dims), kdat.shape[-1]).to(dtype=dtype, device=device)

    # loop over offsets and take advantage of numpy broadcasting
    for Jind in range(nJ):
        curgridind = (kofflist + Jlist[:, Jind].unsqueeze(1)).to(dtype)
        curdistind = torch.round(
            (tm - curgridind) * L.unsqueeze(1)).to(dtype=int_type)
        curgridind = curgridind.to(int_type)

        arr_ind = torch.zeros((M,), dtype=int_type, device=device)
        coef = torch.stack((
            torch.ones(M, dtype=dtype, device=device),
            torch.zeros(M, dtype=dtype, device=device)
        ))

        for d in range(ndims):  # spatial dimension
            coef = conj_complex_mult(
                coef,
                table[d][:, curdistind[d, :] + centers[d]],
                dim=0
            )
            arr_ind = arr_ind + torch.remainder(curgridind[d, :], dims[d]).view(-1) * \
                torch.prod(dims[d + 1:])

        if matadj:
            sparse_coords = torch.stack(
                (
                    arr_ind,
                    torch.arange(
                        arr_ind.shape[0],
                        dtype=arr_ind.dtype,
                        device=arr_ind.device
                    )
                )
            )
            coef_mat_real = coef_mat_real + torch.sparse.FloatTensor(
                sparse_coords,
                coef[0],
                torch.Size((torch.prod(dims), arr_ind.shape[0]))
            )
            coef_mat_imag = coef_mat_imag + torch.sparse.FloatTensor(
                sparse_coords,
                coef[1],
                torch.Size((torch.prod(dims), arr_ind.shape[0]))
            )

        # try to broadcast multiply - batch over coil if not enough memory
        # handle write collisions by using bincount
        if not matadj:
            raise_error = False
            try:
                tmp = complex_mult(coef.unsqueeze(0), kdat, dim=1)

                sparse_coords = torch.stack(
                    (
                        arr_ind,
                        torch.arange(
                            arr_ind.shape[0],
                            dtype=arr_ind.dtype,
                            device=arr_ind.device
                        )
                    )
                )
                sparse_accum_mat = torch.sparse.FloatTensor(
                    sparse_coords,
                    torch.ones(arr_ind.shape[0], dtype=dtype, device=device),
                    torch.Size((torch.prod(dims), arr_ind.shape[0]))
                )

                if coil_broadcast:
                    griddat[:, 0, :] = griddat[:, 0, :] + torch.mm(
                        sparse_accum_mat, tmp[:, 0, :].t().reshape(-1, tmp.shape[0])).t()
                    griddat[:, 1, :] = griddat[:, 1, :] + torch.mm(
                        sparse_accum_mat, tmp[:, 1, :].t().reshape(-1, tmp.shape[0])).t()
                else:
                    for coilind in range(griddat.shape[0]):
                        real_bin = torch.bincount(
                            arr_ind, tmp[coilind, 0, ...])
                        griddat[coilind, 0, :len(real_bin)] = \
                            griddat[coilind, 0, :len(real_bin)] + real_bin
                        imag_bin = torch.bincount(
                            arr_ind, tmp[coilind, 1, ...])
                        griddat[coilind, 1, :len(imag_bin)] = \
                            griddat[coilind, 1, :len(imag_bin)] + imag_bin

            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_error:
                    torch.cuda.empty_cache()
                    for coilind in range(griddat.shape[0]):
                        tmp = complex_mult(coef, kdat[coilind, ...], dim=0)
                        real_bin = torch.bincount(
                            arr_ind, tmp[0, ...])
                        griddat[coilind, 0, :len(real_bin)] = \
                            griddat[coilind, 0, :len(real_bin)] + real_bin
                        imag_bin = torch.bincount(
                            arr_ind, tmp[1, ...])
                        griddat[coilind, 1, :len(imag_bin)] = \
                            griddat[coilind, 1, :len(imag_bin)] + imag_bin
                    raise_error = True
                else:
                    raise e
            except BaseException:
                raise e

    if matadj:
        griddat[:, 0, :] = torch.mm(
            coef_mat_real,
            kdat[:, 0, :].t().reshape(-1, kdat.shape[0])
        ).t() - torch.mm(
            coef_mat_imag,
            kdat[:, 1, :].t().reshape(-1, kdat.shape[0])
        ).t()
        griddat[:, 1, :] = torch.mm(
            coef_mat_real,
            kdat[:, 1, :].t().reshape(-1, kdat.shape[0])
        ).t() + torch.mm(
            coef_mat_imag,
            kdat[:, 0, :].t().reshape(-1, kdat.shape[0])
        ).t()

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
            'real_interp_mats' and 'imag_interp_mats', each key containing
            a list of interpolation matrices (see 
            mri.sparse_interp_mat.precomp_sparse_mats for construction).
            If None, then a standard interpolation is run.
    """
    dtype = interpob['table'][0].dtype
    device = interpob['table'][0].device

    # extract interpolation params and match device and dtype to input
    table = interpob['table']
    n_shift = interpob['n_shift']
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

    y = []
    # run the table interpolator for each batch element
    for b in range(x.shape[0]):
        ncoil = x[b].shape[0]
        if interp_mats is None:
            y.append(
                run_interp(
                    x[b].view((x.shape[1], 2, -1)),
                    torch.tensor(x[b].shape[2:],
                                 dtype=dtype, device=device),
                    table,
                    numpoints,
                    torch.tensor(
                        np.transpose(np.array(Jgen)),
                        dtype=dtype,
                        device=device
                    ),
                    table_oversamp,
                    tm[b],
                    torch.zeros(
                        size=((ncoil, 2, M)),
                        dtype=dtype,
                        device=device
                    )
                )
            )
        else:
            # moving interp mats to device is a little ugly, but probably
            # worth it as it prevents a lot of silly bugs from being
            # consequential
            y.append(
                run_mat_interp(
                    x[b].view((x.shape[1], 2, -1)),
                    interp_mats['real_interp_mats'][b].to(
                        dtype=dtype, device=device),
                    interp_mats['imag_interp_mats'][b].to(
                        dtype=dtype, device=device),
                    torch.zeros(
                        size=((ncoil, 2, M)),
                        dtype=dtype,
                        device=device
                    )
                )
            )

        # phase for fftshift
        phase = imag_exp(
            torch.mv(
                torch.transpose(om[b], 1, 0),
                n_shift
            )
        ).unsqueeze(0)
        y[-1] = complex_mult(y[-1], phase, dim=1)

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
            'real_interp_mats' and 'imag_interp_mats', each key containing
            a list of interpolation matrices (see 
            mri.sparse_interp_mat.precomp_sparse_mats for construction).
            If None, then a standard interpolation is run.
    """
    y = y.clone()

    table = interpob['table']
    n_shift = interpob['n_shift']
    grid_size = interpob['grid_size']
    numpoints = interpob['numpoints']
    table_oversamp = interpob['table_oversamp']
    coil_broadcast = interpob['coil_broadcast']
    matadj = interpob['matadj']

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

    x = []
    # run the table interpolator for each batch element
    for b in range(y.shape[0]):
        ncoil = y[b].shape[0]

        # phase for fftshift
        phase = imag_exp(
            torch.mv(
                torch.transpose(om[b], 1, 0),
                n_shift
            )
        )
        y[b] = conj_complex_mult(y[b], phase.unsqueeze(0), dim=1)

        if interp_mats is None:
            x.append(
                run_interp_back(
                    y[b],
                    grid_size,
                    table,
                    numpoints,
                    torch.tensor(
                        np.transpose(np.array(Jgen)),
                        dtype=dtype,
                        device=device
                    ),
                    table_oversamp,
                    tm[b],
                    torch.zeros(
                        size=(ncoil, 2, np.prod(
                            np.array(grid_size.cpu())).astype(np.int)),
                        dtype=dtype,
                        device=device
                    ),
                    coil_broadcast=coil_broadcast,
                    matadj=matadj
                )
            )
        else:
            # moving interp mats to device is a little ugly, but probably
            # worth it as it prevents a lot of silly bugs from being
            # consequential
            x.append(
                run_mat_interp_back(
                    y[b],
                    interp_mats['real_interp_mats'][b].to(
                        dtype=dtype, device=device),
                    interp_mats['imag_interp_mats'][b].to(
                        dtype=dtype, device=device),
                    torch.zeros(
                        size=(ncoil, 2, np.prod(
                            np.array(grid_size.cpu())).astype(np.int)),
                        dtype=dtype,
                        device=device
                    )
                )
            )

    x = torch.stack(x)

    bsize = y.shape[0]
    ncoil = y.shape[1]
    out_size = (bsize, ncoil, 2) + \
        tuple(np.array(grid_size.cpu()).astype(np.int))

    x = x.view(out_size)

    return x
