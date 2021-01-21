from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .._math import complex_mult, conj_complex_mult, imag_exp


def spmat_interp(
    image: Tensor, interp_mats: Union[Tensor, Tuple[Tensor, Tensor]]
) -> Tensor:
    """Sparse matrix interpolation backend."""
    if not isinstance(interp_mats, tuple):
        raise TypeError("interp_mats must be 2-tuple of (real_mat, imag_mat.")

    coef_mat_real, coef_mat_imag = interp_mats
    batch_size, num_coils = image.shape[:2]

    if image.is_complex():
        image = torch.view_as_real(image)
        output_size = [batch_size, num_coils, -1]
        is_complex = True
    else:
        output_size = [batch_size, num_coils, -1, 2]
        is_complex = False

    # we have to do these transposes because torch.mm requires first to be spmatrix
    image = image.reshape(batch_size * num_coils, -1, 2)
    real_griddat = image.select(-1, 0).t()
    imag_griddat = image.select(-1, 1).t()

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

    if is_complex:
        kdat = torch.view_as_complex(kdat)

    return kdat.view(*output_size)


def spmat_interp_adjoint(
    data: Tensor,
    interp_mats: Union[Tensor, Tuple[Tensor, Tensor]],
    grid_size: Tensor,
) -> Tensor:
    """Sparse matrix interpolation adjoint backend."""
    if not isinstance(interp_mats, tuple):
        raise TypeError("interp_mats must be 2-tuple of (real_mat, imag_mat.")

    coef_mat_real, coef_mat_imag = interp_mats
    batch_size, num_coils = data.shape[:2]

    if data.is_complex():
        data = torch.view_as_real(data)
        output_size = [batch_size, num_coils] + grid_size.tolist()
        is_complex = True
    else:
        output_size = [batch_size, num_coils] + grid_size.tolist() + [2]
        is_complex = False

    # we have to do these transposes because torch.mm requires first to be spmatrix
    real_kdat = data.select(-1, 0).view(-1, data.shape[-2]).t()
    imag_kdat = data.select(-1, 1).view(-1, data.shape[-2]).t()
    coef_mat_real = coef_mat_real.t()
    coef_mat_imag = coef_mat_imag.t()

    # apply multiplies with complex conjugate
    image = torch.stack(
        [
            (
                torch.mm(coef_mat_real, real_kdat) + torch.mm(coef_mat_imag, imag_kdat)
            ).t(),
            (
                torch.mm(coef_mat_real, imag_kdat) - torch.mm(coef_mat_imag, real_kdat)
            ).t(),
        ],
        dim=-1,
    )

    if is_complex:
        image = torch.view_as_complex(image)

    return image.reshape(*output_size)


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
        tables: A list of tensors tabulating a Kaiser-Bessel interpolation
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

        arr_ind = arr_ind + torch.remainder(it_gridind, it_grid_size).view(
            -1
        ) * torch.prod(grid_size[d + 1 :])

    return coef, arr_ind


@torch.jit.script
def table_interp(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Table interpolation backend."""
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


@torch.jit.script
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
    """Table interpolation adjoint backend."""
    dtype = data.dtype
    device = data.device
    int_type = torch.long

    output_prod = int(torch.prod(grid_size))
    output_size = [data.shape[0], data.shape[1]]
    for el in grid_size:
        output_size.append(int(el))

    if torch.is_complex(data):
        is_complex = True
    else:
        is_complex = False
        output_size.append(2)
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
                size=(mini_data.shape[0], output_prod),
                dtype=dtype,
                device=device,
            )
        else:
            image = torch.zeros(
                size=(mini_data.shape[0], output_prod, 2),
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

            image.index_add_(1, arr_ind, complex_mult(coef, mini_data))

        output.append(image)

    return torch.stack(output).reshape(output_size)
