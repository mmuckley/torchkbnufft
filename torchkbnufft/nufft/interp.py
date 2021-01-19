from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from ..math import complex_mult, conj_complex_mult, imag_exp


def spmat_interp(
    griddat: Tensor, interp_mats: Union[Tensor, Tuple[Tensor, Tensor]]
) -> Tensor:
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
    if isinstance(interp_mats, tuple):
        is_complex = False
        coef_mat_real, coef_mat_imag = interp_mats
        if not (griddat.dtype == coef_mat_real.dtype == coef_mat_imag.dtype):
            raise ValueError("Non-matching dtypes.")
    elif torch.is_complex(interp_mats):
        if not griddat.dtype == interp_mats.dtype:
            raise ValueError("Non-matching dtypes.")
        is_complex = True
    else:
        raise ValueError("Variable interp_mats must be complex if not tuple.")

    batch_size, num_coils = griddat.shape[:2]

    if is_complex:
        assert isinstance(interp_mats, Tensor)
        output_size = [batch_size, num_coils, -1]

        griddat = griddat.view(batch_size * num_coils, -1)
        kdat = torch.mm(interp_mats, griddat.t()).t()
    else:
        output_size = [batch_size, num_coils, -1, 2]

        # we have to do these transposes because torch.mm requires first to be spmatrix
        griddat = griddat.view(batch_size * num_coils, -1, 2)
        real_griddat = griddat.select(-1, 0).t()
        imag_griddat = griddat.select(-1, 1).t()

        # apply multiplies
        kdat = torch.stack(
            [
                (
                    torch.mm(coef_mat_real, real_griddat)
                    - torch.mm(coef_mat_imag, imag_griddat)
                ).t(),
                (
                    torch.mm(coef_mat_real, imag_griddat)
                    + torch.mm(coef_mat_imag, real_griddat)
                ).t(),
            ],
            dim=-1,
        )

    return kdat.view(*output_size)


def spmat_interp_adjoint(
    kdat: Tensor,
    interp_mats: Union[Tensor, Tuple[Tensor, Tensor]],
    grid_size: Tensor,
) -> Tensor:
    """Interpolates kdat to on-grid coordinates with input sparse matrices.

    Args:
        kdat: The off-grid frequency data.
        interp_mats: Sparse interpolation matrices.
        grid_size: The size of the output grid.

    Returns:
        kdat interpolated to on-grid locations.
    """
    if isinstance(interp_mats, tuple):
        is_complex = False
        coef_mat_real, coef_mat_imag = interp_mats
        if not (kdat.dtype == coef_mat_real.dtype == coef_mat_imag.dtype):
            raise ValueError("Non-matching dtypes.")
    elif torch.is_complex(interp_mats):
        if not kdat.dtype == interp_mats.dtype:
            raise ValueError("Non-matching dtypes.")
        is_complex = True
    else:
        raise ValueError("Variable interp_mats must be complex if not tuple.")

    batch_size, num_coils = kdat.shape[:2]

    if is_complex:
        assert isinstance(interp_mats, Tensor)
        output_size = [batch_size, num_coils] + grid_size.tolist()

        griddat = torch.mm(interp_mats, kdat.t())
    else:
        output_size = [batch_size, num_coils] + grid_size.tolist() + [2]

        # we have to do these transposes because torch.mm requires first to be spmatrix
        real_kdat = kdat.select(-1, 0).t().view(-1, kdat.shape[0])
        imag_kdat = kdat.select(-1, 1).t().view(-1, kdat.shape[0])
        coef_mat_real = coef_mat_real.t_()
        coef_mat_imag = coef_mat_imag.t_()

        # apply multiplies with complex conjugate
        griddat = torch.stack(
            [
                (
                    torch.mm(coef_mat_real, real_kdat)
                    + torch.mm(coef_mat_imag, imag_kdat)
                ).t(),
                (
                    torch.mm(coef_mat_real, imag_kdat)
                    - torch.mm(coef_mat_imag, real_kdat)
                ).t(),
            ],
            dim=-1,
        )

        # put the matrices back in the order we were given
        coef_mat_real = coef_mat_real.t_()
        coef_mat_imag = coef_mat_imag.t_()

    return griddat.view(*output_size)


def calc_coef_and_indices(
    tm: Tensor,
    base_offset: Tensor,
    offset_increments: Tensor,
    tables: List[Tensor],
    centers: Tensor,
    table_oversamp: Tensor,
    grid_size: Tensor,
    conjcoef: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Calculates interpolation coefficients and on-grid indices.

    Args:
        tm: Normalized frequency locations.
        base_offset: A tensor with offset locations to first elements in list
            of nearest neighbors.
        offset_increments: A tensor for how much to increment offsets.
        table: A list of tensors tabulating a Kaiser-Bessel interpolation
            kernel.
        centers: A tensor with the center locations of the table for each
            dimension.
        table_oversamp: A tensor with the table size in each dimension.
        grid_size: A tensor with image dimensions.
        conjcoef: A boolean for whether to compute normal or complex conjugate
            interpolation coefficients (conjugate needed for adjoint).

    Returns:
        A tuple with interpolation coefficients and indices.
    """
    assert len(tables) == len(offset_increments)
    assert len(tables) == len(centers)

    # type values
    dtype = tables[0].dtype
    device = tm.device
    int_type = torch.long

    ktraj_len = tm.shape[1]

    # indexing locations
    gridind = base_offset + offset_increments.unsqueeze(1)
    distind = torch.round((tm - gridind.to(tm)) * table_oversamp.unsqueeze(1)).to(
        dtype=int_type
    )
    arr_ind = torch.zeros(ktraj_len, dtype=int_type, device=device)

    # give complex numbers if requested
    coef = torch.ones(ktraj_len, dtype=dtype, device=device)
    if not torch.is_complex(coef):
        coef = torch.stack((coef, torch.zeros_like(coef)), dim=-1)

    for d, (table, it_distind, center, it_gridind, it_grid_size) in enumerate(
        zip(tables, distind, centers, gridind, grid_size)
    ):  # spatial dimension
        if conjcoef:
            coef = conj_complex_mult(coef, table[it_distind + center])
        else:
            coef = complex_mult(coef, table[it_distind + center])

        arr_ind = arr_ind + torch.remainder(it_gridind, it_grid_size).view(-1)
        arr_ind = arr_ind * torch.prod(grid_size[d + 1 :])

    return coef, arr_ind


def table_interp(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Apply table interpolation.

    Inputs are assumed to be batch/chans x coil x image dims x real/imag.
    Om should be nbatch x ndims x klength.

    Args:
        x (tensor): The oversampled DFT of the signal.
        om (tensor, optional): A custom set of k-space points to
            interpolate to in radians/voxel.
        interpob (dict): An interpolation object with 'table', 'n_shift',
            'grid_size', 'numpoints', and 'table_oversamp' keys. See
            models.kbinterp.py for details.

    Returns:
        tensor: The signal interpolated to off-grid locations.
    """
    dtype = image.dtype
    device = image.device
    int_type = torch.long

    if torch.is_complex(image):
        is_complex = True
        grid_size = torch.tensor(image.shape[2:], dtype=int_type, device=device)
    else:
        is_complex = False
        grid_size = torch.tensor(image.shape[2:-1], dtype=int_type, device=device)
        tables = [torch.view_as_real(table) for table in tables]

    # convert to normalized freq locs
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))

    # compute interpolation centers
    centers = torch.floor(numpoints * table_oversamp / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    base_offset = 1 + torch.floor(tm - numpoints.unsqueeze(1) / 2.0).to(dtype=int_type)

    # run the table interpolator for each batch element
    output = []
    for mini_image in image:
        # flatten image dimensions, initialize output
        if is_complex:
            mini_image = mini_image.reshape(mini_image.shape[0], -1)
            kdat = torch.zeros(
                size=(mini_image.shape[0], tm.shape[-1]), dtype=dtype, device=device
            )
        else:
            mini_image = mini_image.reshape(mini_image.shape[0], -1, 2)
            kdat = torch.zeros(
                size=((mini_image.shape[0], tm.shape[-1], 2)),
                dtype=dtype,
                device=device,
            )

        # loop over offsets and take advantage of broadcasting
        for offset in offsets:
            coef, arr_ind = calc_coef_and_indices(
                tm=tm,
                base_offset=base_offset,
                offset_increments=offset,
                tables=tables,
                centers=centers,
                table_oversamp=table_oversamp,
                grid_size=grid_size,
            )

            # unsqueeze coil dimension for on-grid indices
            arr_ind = arr_ind.unsqueeze(0).expand(kdat.shape[0], -1)

            # gather and multiply coefficients
            if is_complex:
                kdat += complex_mult(
                    coef.unsqueeze(0), torch.gather(mini_image, 1, arr_ind)
                )
            else:
                kdat += complex_mult(
                    coef.unsqueeze(0),
                    torch.stack(
                        (
                            torch.gather(mini_image.select(-1, 0), 1, arr_ind),
                            torch.gather(mini_image.select(-1, 1), 1, arr_ind),
                        ),
                        dim=-1,
                    ),
                )

        # phase for fftshift
        kdat = complex_mult(
            kdat,
            imag_exp(
                torch.mv(torch.transpose(omega, 1, 0), n_shift),
                return_complex=is_complex,
            ).unsqueeze(0),
        )

        output.append(kdat)

    return torch.stack(output)


def table_interp_adjoint(
    data: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    grid_size: Tensor,
) -> Tensor:
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
    dtype = data.dtype
    device = data.device
    int_type = torch.long

    if torch.is_complex(data):
        is_complex = True
        output_size = [data.shape[0], data.shape[1]] + grid_size.tolist()
    else:
        is_complex = False
        output_size = [data.shape[0], data.shape[1]] + grid_size.tolist() + [2]
        tables = [torch.view_as_real(table) for table in tables]

    # convert to normalized freq locs
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))

    # compute interpolation centers
    centers = torch.floor(numpoints * table_oversamp / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    base_offset = 1 + torch.floor(tm - numpoints.unsqueeze(1) / 2.0).to(dtype=int_type)

    # run the table interpolator for each batch element
    output = []
    for mini_data in data:
        if is_complex:
            image = torch.zeros(
                size=(mini_data.shape[0], torch.prod(grid_size)),
                dtype=dtype,
                device=device,
            )
        else:
            image = torch.zeros(
                size=(mini_data.shape[0], torch.prod(grid_size), 2),
                dtype=dtype,
                device=device,
            )

        # phase for fftshift
        mini_data = conj_complex_mult(
            mini_data,
            imag_exp(
                torch.mv(torch.transpose(omega, 1, 0), n_shift),
                return_complex=is_complex,
            ),
        )

        # loop over offsets and take advantage of numpy broadcasting
        for offset in offsets:
            coef, arr_ind = calc_coef_and_indices(
                tm=tm,
                base_offset=base_offset,
                offset_increments=offset,
                tables=tables,
                centers=centers,
                table_oversamp=table_oversamp,
                grid_size=grid_size,
                conjcoef=True,
            )

            image.index_add_(2, arr_ind, complex_mult(coef, mini_data))

        output.append(image)

    return torch.stack(output).reshape(output_size)
