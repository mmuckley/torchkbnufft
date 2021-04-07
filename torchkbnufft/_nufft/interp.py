from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .._math import imag_exp

# a little hacky but we don't have a function for detecting OMP
USING_OMP = "USE_OPENMP=ON" in torch.__config__.show()


def spmat_interp(
    image: Tensor, interp_mats: Union[Tensor, Tuple[Tensor, Tensor]]
) -> Tensor:
    """Sparse matrix interpolation backend."""
    if not isinstance(interp_mats, tuple):
        raise TypeError("interp_mats must be 2-tuple of (real_mat, imag_mat.")

    coef_mat_real, coef_mat_imag = interp_mats
    batch_size, num_coils = image.shape[:2]

    # sparse matrix multiply requires real
    image = torch.view_as_real(image)
    output_size = [batch_size, num_coils, -1]

    # we have to do these transposes because torch.mm requires first to be spmatrix
    image = image.reshape(batch_size * num_coils, -1, 2)
    real_griddat = image.select(-1, 0).t().contiguous()
    imag_griddat = image.select(-1, 1).t().contiguous()

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

    return torch.view_as_complex(kdat).reshape(*output_size)


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

    # sparse matrix multiply requires real
    data = torch.view_as_real(data)
    output_size = [batch_size, num_coils] + grid_size.tolist()

    # we have to do these transposes because torch.mm requires first to be spmatrix
    real_kdat = data.select(-1, 0).view(-1, data.shape[-2]).t().contiguous()
    imag_kdat = data.select(-1, 1).view(-1, data.shape[-2]).t().contiguous()
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

    return torch.view_as_complex(image).reshape(*output_size)


@torch.jit.script
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

    for d, (table, it_distind, center, it_gridind, it_grid_size) in enumerate(
        zip(tables, distind, centers, gridind, grid_size)
    ):  # spatial dimension
        if conjcoef:
            coef = coef * table[it_distind + center].conj()
        else:
            coef = coef * table[it_distind + center]

        arr_ind = arr_ind + torch.remainder(it_gridind, it_grid_size).view(
            -1
        ) * torch.prod(grid_size[d + 1 :])

    return coef, arr_ind


@torch.jit.script
def table_interp_one_batch(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Table interpolation backend (see ``table_interp()``)."""
    dtype = image.dtype
    device = image.device
    int_type = torch.long

    grid_size = torch.tensor(image.shape[2:], dtype=int_type, device=device)

    # convert to normalized freq locs
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))

    # compute interpolation centers
    centers = torch.floor(numpoints * table_oversamp / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    base_offset = 1 + torch.floor(tm - numpoints.unsqueeze(-1) / 2.0).to(dtype=int_type)

    # flatten image dimensions
    image = image.reshape(image.shape[0], image.shape[1], -1)
    kdat = torch.zeros(
        image.shape[0], image.shape[1], tm.shape[-1], dtype=dtype, device=device
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

        # gather and multiply coefficients
        kdat += coef * image[:, :, arr_ind]

    # phase for fftshift
    return kdat * imag_exp(
        torch.sum(omega * n_shift.unsqueeze(-1), dim=-2, keepdim=True),
        return_complex=True,
    )


@torch.jit.script
def table_interp_multiple_batches(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
) -> Tensor:
    """Table interpolation with for loop over batch dimension."""
    kdat = []
    for (it_image, it_omega) in zip(image, omega):
        kdat.append(
            table_interp_one_batch(
                it_image.unsqueeze(0),
                it_omega,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    return torch.cat(kdat)


@torch.jit.script
def table_interp_fork_over_batchdim(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    num_forks: int,
) -> Tensor:
    """Table interpolation with forking over k-space."""
    # initialize the fork processes
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for (image_chunk, omega_chunk) in zip(
        image.tensor_split(num_forks), omega.tensor_split(num_forks)
    ):
        futures.append(
            torch.jit.fork(
                table_interp_multiple_batches,
                image_chunk,
                omega_chunk,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    # collect the results
    return torch.cat([torch.jit.wait(future) for future in futures])


@torch.jit.script
def table_interp_fork_over_kspace(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    num_forks: int,
) -> Tensor:
    """Table interpolation backend (see table_interp())."""
    # indexing is worst when we have repeated indices - let's spread them out
    klength = omega.shape[1]
    omega_chunks = [omega[:, ind:klength:num_forks] for ind in range(num_forks)]

    # initialize the fork processes
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for omega_chunk in omega_chunks:
        futures.append(
            torch.jit.fork(
                table_interp_one_batch,
                image,
                omega_chunk,
                tables,
                n_shift,
                numpoints,
                table_oversamp,
                offsets,
            )
        )

    kdat = torch.zeros(
        image.shape[0],
        image.shape[1],
        omega.shape[1],
        dtype=image.dtype,
        device=image.device,
    )

    # collect the results
    for ind, future in enumerate(futures):
        kdat[:, :, ind:klength:num_forks] = torch.jit.wait(future)

    return kdat


def table_interp(
    image: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    min_kspace_per_fork: int = 1024,
) -> Tensor:
    """Table interpolation backend.

    This interpolates from a gridded set of data to off-grid of data given by
    the coordinates in ``omega``.

    Args:
        image: Gridded data to interpolate from.
        omega: Fourier coordinates to interpolate to (in radians/voxel, -pi to
            pi).
        tables: List of tables for each image dimension.
        n_shift: Size of desired fftshift.
        numpoints: Number of neighbors in each dimension.
        table_oversamp: Size of table in each dimension.
        offsets: A list of offset values for interpolation.
        min_kspace_per_fork: Minimum number of k-space samples to use in each
            process fork. Only used for single trajectory on CPU.

    Returns:
        ``image`` interpolated to k-space locations at ``omega``.
    """
    if omega.ndim not in (2, 3):
        raise ValueError("omega must have 2 or 3 dimensions.")

    if omega.ndim == 3:
        if omega.shape[0] == 1:
            omega = omega[0]  # broadcast a single traj

    if omega.ndim == 3:
        if not omega.shape[0] == image.shape[0]:
            raise ValueError(
                "If omega has batch dim, omega batch dimension must match image."
            )

    # we fork processes for accumulation, so we need to do a bit of thread
    # management for OMP to make sure we don't oversubscribe (managment not
    # necessary for non-OMP)
    num_threads = torch.get_num_threads()
    factors = torch.arange(1, num_threads + 1)
    factors = factors[torch.remainder(torch.tensor(num_threads), factors) == 0]
    threads_per_fork = num_threads  # default fallback

    if omega.ndim == 3:
        # increase number of forks as long as it's not greater than batch size
        for factor in factors.flip(0):
            if num_threads // factor <= omega.shape[0]:
                threads_per_fork = int(factor)

        num_forks = num_threads // threads_per_fork

        if USING_OMP and image.device == torch.device("cpu"):
            torch.set_num_threads(threads_per_fork)
        kdat = table_interp_fork_over_batchdim(
            image, omega, tables, n_shift, numpoints, table_oversamp, offsets, num_forks
        )
        if USING_OMP and image.device == torch.device("cpu"):
            torch.set_num_threads(num_threads)
    elif image.device == torch.device("cpu"):
        # determine number of process forks while keeping a minimum amount of
        # k-space per fork
        for factor in factors.flip(0):
            if omega.shape[1] / (num_threads // factor) >= min_kspace_per_fork:
                threads_per_fork = int(factor)

        num_forks = num_threads // threads_per_fork

        if USING_OMP:
            torch.set_num_threads(threads_per_fork)
        kdat = table_interp_fork_over_kspace(
            image, omega, tables, n_shift, numpoints, table_oversamp, offsets, num_forks
        )
        if USING_OMP:
            torch.set_num_threads(num_threads)
    else:
        # no forking for batchless omega on GPU
        kdat = table_interp_one_batch(
            image, omega, tables, n_shift, numpoints, table_oversamp, offsets
        )

    return kdat


@torch.jit.script
def accum_tensor_index_add(image: Tensor, arr_ind: Tensor, data: Tensor) -> Tensor:
    """We fork this function for the adjoint accumulation."""
    if arr_ind.ndim == 2:
        for (image_batch, arr_ind_batch, data_batch) in zip(image, arr_ind, data):
            for (image_coil, data_coil) in zip(image_batch, data_batch):
                image_coil.index_add_(0, arr_ind_batch, data_coil)
    else:
        for (image_it, data_it) in zip(image, data):
            image_it.index_add_(0, arr_ind, data_it)

    return image


@torch.jit.script
def fork_and_accum(
    image: Tensor, arr_ind: Tensor, data: Tensor, num_forks: int
) -> Tensor:
    """Process forking and per batch/coil accumulation function."""
    # initialize the fork processes
    futures: List[torch.jit.Future[torch.Tensor]] = []
    if arr_ind.ndim == 2:
        for (image_chunk, arr_ind_chunk, data_chunk) in zip(
            image.tensor_split(num_forks),
            arr_ind.tensor_split(num_forks),
            data.tensor_split(num_forks),
        ):
            futures.append(
                torch.jit.fork(
                    accum_tensor_index_add,
                    image_chunk,
                    arr_ind_chunk,
                    data_chunk,
                )
            )
    else:
        for (image_chunk, data_chunk) in zip(
            image.tensor_split(num_forks), data.tensor_split(num_forks)
        ):
            futures.append(
                torch.jit.fork(
                    accum_tensor_index_add,
                    image_chunk,
                    arr_ind,
                    data_chunk,
                )
            )

    # wait for processes to finish
    # results in-place
    _ = [torch.jit.wait(future) for future in futures]

    return image


@torch.jit.script
def calc_coef_and_indices_batch(
    tm: Tensor,
    base_offset: Tensor,
    offset_increments: Tensor,
    tables: List[Tensor],
    centers: Tensor,
    table_oversamp: Tensor,
    grid_size: Tensor,
    conjcoef: bool,
) -> Tuple[Tensor, Tensor]:
    """For loop coef calculation over batch dim."""
    coef = []
    arr_ind = []
    for (tm_it, base_offset_it) in zip(tm, base_offset):
        coef_it, arr_ind_it = calc_coef_and_indices(
            tm=tm_it,
            base_offset=base_offset_it,
            offset_increments=offset_increments,
            tables=tables,
            centers=centers,
            table_oversamp=table_oversamp,
            grid_size=grid_size,
            conjcoef=conjcoef,
        )

        coef.append(coef_it)
        arr_ind.append(arr_ind_it)

    return (torch.stack(coef), torch.stack(arr_ind))


@torch.jit.script
def calc_coef_and_indices_fork_over_batches(
    tm: Tensor,
    base_offset: Tensor,
    offset_increments: Tensor,
    tables: List[Tensor],
    centers: Tensor,
    table_oversamp: Tensor,
    grid_size: Tensor,
    conjcoef: bool,
    num_forks: int,
) -> Tuple[Tensor, Tensor]:
    """Split work across batchdim, fork processes."""
    if tm.ndim == 3:
        if tm.shape[0] == 1:
            tm = tm[0]

    if tm.ndim == 2:
        coef, arr_ind = calc_coef_and_indices(
            tm=tm,
            base_offset=base_offset,
            offset_increments=offset_increments,
            tables=tables,
            centers=centers,
            table_oversamp=table_oversamp,
            grid_size=grid_size,
            conjcoef=conjcoef,
        )
    else:
        # initialize the fork processes
        futures: List[torch.jit.Future[Tuple[Tensor, Tensor]]] = []
        for (tm_chunk, base_offset_chunk) in zip(
            tm.tensor_split(num_forks),
            base_offset.tensor_split(num_forks),
        ):
            futures.append(
                torch.jit.fork(
                    calc_coef_and_indices_batch,
                    tm_chunk,
                    base_offset_chunk,
                    offset_increments,
                    tables,
                    centers,
                    table_oversamp,
                    grid_size,
                    conjcoef,
                )
            )

        # collect the results
        results = [torch.jit.wait(future) for future in futures]
        coef = torch.cat([result[0] for result in results])
        arr_ind = torch.cat([result[1] for result in results])

    return coef, arr_ind


@torch.jit.script
def sort_one_batch(
    tm: Tensor, omega: Tensor, data: Tensor, grid_size: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sort input tensors by ordered values of tm."""
    tmp = torch.zeros(omega.shape[1], dtype=omega.dtype, device=omega.device)
    for d, dim in enumerate(grid_size):
        tmp = tmp + torch.remainder(tm[d], dim) * torch.prod(grid_size[d + 1 :])

    _, indices = torch.sort(tmp)

    return tm[:, indices], omega[:, indices], data[:, :, indices]


@torch.jit.script
def sort_data(
    tm: Tensor, omega: Tensor, data: Tensor, grid_size: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sort input tensors by ordered values of tm."""
    if omega.ndim == 3:
        # loop over batch dimension to get sorted k-space
        results: List[Tuple[Tensor, Tensor, Tensor]] = []
        for (tm_it, omega_it, data_it) in zip(tm, omega, data):
            results.append(
                sort_one_batch(tm_it, omega_it, data_it.unsqueeze(0), grid_size)
            )

        tm_ret = torch.stack([result[0] for result in results])
        omega_ret = torch.stack([result[1] for result in results])
        data_ret = torch.cat([result[2] for result in results])
    else:
        tm_ret, omega_ret, data_ret = sort_one_batch(tm, omega, data, grid_size)

    return tm_ret, omega_ret, data_ret


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
    """Table interpolation adjoint backend.

    This interpolates from an off-grid set of data at coordinates given by
    ``omega`` to on-grid locations.

    Args:
        data: Off-grid data to interpolate from.
        omega: Fourier coordinates to interpolate to (in radians/voxel, -pi to
            pi).
        tables: List of tables for each image dimension.
        n_shift: Size of desired fftshift.
        numpoints: Number of neighbors in each dimension.
        table_oversamp: Size of table in each dimension.
        offsets: A list of offset values for interpolation.
        grid_size: Size of grid to interpolate to.

    Returns:
        ``data`` interpolated to gridded locations.
    """
    if omega.ndim not in (2, 3):
        raise ValueError("omega must have 2 or 3 dimensions.")

    if omega.ndim == 3:
        if omega.shape[0] == 1:
            omega = omega[0]  # broadcast a single traj

    if omega.ndim == 3:
        if not omega.shape[0] == data.shape[0]:
            raise ValueError(
                "If omega has batch dim, omega batch dimension must match data."
            )

    dtype = data.dtype
    device = data.device
    int_type = torch.long

    # we fork processes for accumulation, so we need to do a bit of thread
    # management for OMP to make sure we don't oversubscribe (managment not
    # necessary for non-OMP)
    num_threads = torch.get_num_threads()
    factors = torch.arange(1, num_threads + 1)
    factors = factors[torch.remainder(torch.tensor(num_threads), factors) == 0]
    threads_per_fork = num_threads  # default fallback

    if omega.ndim == 3:
        # increase number of forks as long as it's not greater than batch size
        for factor in factors.flip(0):
            if num_threads // factor <= omega.shape[0]:
                threads_per_fork = int(factor)
    else:
        # increase forks as long as it's less/eq than batch * coildim
        for factor in factors.flip(0):
            if num_threads // factor <= data.shape[0] * data.shape[1]:
                threads_per_fork = int(factor)

    num_forks = num_threads // threads_per_fork

    # calculate output size
    output_prod = int(torch.prod(grid_size))
    output_size = [data.shape[0], data.shape[1]]
    for el in grid_size:
        output_size.append(int(el))

    # convert to normalized freq locs and sort
    tm = omega / (2 * np.pi / grid_size.to(omega).unsqueeze(-1))
    tm, omega, data = sort_data(tm, omega, data, grid_size)

    # compute interpolation centers
    centers = torch.floor(numpoints * table_oversamp / 2).to(dtype=int_type)

    # offset from k-space to first coef loc
    base_offset = 1 + torch.floor(tm - numpoints.unsqueeze(-1) / 2.0).to(dtype=int_type)

    # initialized flattened image
    image = torch.zeros(
        size=(data.shape[0], data.shape[1], output_prod),
        dtype=dtype,
        device=device,
    )

    # phase for fftshift
    data = (
        data
        * imag_exp(
            torch.sum(omega * n_shift.unsqueeze(-1), dim=-2, keepdim=True),
            return_complex=True,
        ).conj()
    )

    # loop over offsets
    for offset in offsets:
        if USING_OMP and device == torch.device("cpu") and tm.ndim == 3:
            torch.set_num_threads(threads_per_fork)
        coef, arr_ind = calc_coef_and_indices_fork_over_batches(
            tm=tm,
            base_offset=base_offset,
            offset_increments=offset,
            tables=tables,
            centers=centers,
            table_oversamp=table_oversamp,
            grid_size=grid_size,
            conjcoef=True,
            num_forks=num_forks,
        )
        if USING_OMP and device == torch.device("cpu") and tm.ndim == 3:
            torch.set_num_threads(num_threads)

        # multiply coefs to data
        if coef.ndim == 2:
            coef = coef.unsqueeze(1)
            assert coef.ndim == data.ndim

        if USING_OMP and device == torch.device("cpu"):
            torch.set_num_threads(threads_per_fork)
        # this is a much faster way of doing index accumulation
        if arr_ind.ndim == 1:
            # fork over coils and batches
            image = fork_and_accum(
                image.view(data.shape[0] * data.shape[1], output_prod),
                arr_ind,
                (coef * data).view(data.shape[0] * data.shape[1], -1),
                num_forks,
            ).view(data.shape[0], data.shape[1], output_prod)
        else:
            # fork just over batches
            image = fork_and_accum(image, arr_ind, coef * data, num_forks)

        if USING_OMP and device == torch.device("cpu"):
            torch.set_num_threads(num_threads)

    return image.view(output_size)
