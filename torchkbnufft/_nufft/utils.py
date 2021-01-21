import itertools
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import special
from scipy.sparse import coo_matrix
from torch import Tensor

DTYPE_MAP = [
    (torch.complex128, torch.float64),
    (torch.complex64, torch.float32),
    (torch.complex32, torch.float16),
]


def build_tensor_spmatrix(
    omega: np.ndarray,
    numpoints: Sequence[int],
    im_size: Sequence[int],
    grid_size: Sequence[int],
    n_shift: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> Tuple[Tensor, Tensor]:
    """Builds a sparse matrix with the interpolation coefficients.

    This builds the interpolation matrices directly from scipy Kaiser-Bessel
    functions, so using them for a NUFFT should be a little more accurate than
    table interpolation.

    Args:
        omega: An array of coordinates to interpolate to (radians/voxel).
        numpoints: Number of points to use for interpolation in each dimension.
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        n_shift: Number of points to shift for fftshifts.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        2-Tuple of (real, imaginary) tensors for NUFFT interpolation.
    """
    coo = build_numpy_spmatrix(
        omega=omega,
        numpoints=numpoints,
        im_size=im_size,
        grid_size=grid_size,
        n_shift=n_shift,
        order=order,
        alpha=alpha,
    )

    values = coo.data
    indices = np.stack((coo.row, coo.col))

    inds = torch.tensor(indices, dtype=torch.long)
    real_vals = torch.tensor(np.real(values))
    imag_vals = torch.tensor(np.imag(values))
    shape = coo.shape

    interp_mats = (
        torch.sparse.FloatTensor(inds, real_vals, torch.Size(shape)),  # type: ignore
        torch.sparse.FloatTensor(inds, imag_vals, torch.Size(shape)),  # type: ignore
    )

    return interp_mats


def build_numpy_spmatrix(
    omega: np.ndarray,
    numpoints: Sequence[int],
    im_size: Sequence[int],
    grid_size: Sequence[int],
    n_shift: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> coo_matrix:
    """Builds a sparse matrix with the interpolation coefficients.

    Args:
        omega: An array of coordinates to interpolate to (radians/voxel).
        numpoints: Number of points to use for interpolation in each dimension.
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        n_shift: Number of points to shift for fftshifts.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        A scipy sparse interpolation matrix.
    """
    spmat = -1

    ndims = omega.shape[0]
    klength = omega.shape[1]

    # calculate interpolation coefficients using kb kernel
    def interp_coeff(om, npts, grdsz, alpha, order):
        gam = 2 * np.pi / grdsz
        interp_dist = om / gam - np.floor(om / gam - npts / 2)
        Jvec = np.reshape(np.array(range(1, npts + 1)), (1, npts))
        kern_in = -1 * Jvec + np.expand_dims(interp_dist, 1)

        cur_coeff = np.zeros(shape=kern_in.shape, dtype=np.complex)
        indices = np.absolute(kern_in) < npts / 2
        bess_arg = np.sqrt(1 - (kern_in[indices] / (npts / 2)) ** 2)
        denom = special.iv(order, alpha)
        cur_coeff[indices] = special.iv(order, alpha * bess_arg) / denom
        cur_coeff = np.real(cur_coeff)

        return cur_coeff, kern_in

    full_coef = []
    kd = []
    for (
        it_om,
        it_im_size,
        it_grid_size,
        it_numpoints,
        it_om,
        it_alpha,
        it_order,
    ) in zip(omega, im_size, grid_size, numpoints, omega, alpha, order):
        # get the interpolation coefficients
        coef, kern_in = interp_coeff(
            it_om, it_numpoints, it_grid_size, it_alpha, it_order
        )

        gam = 2 * np.pi / it_grid_size
        phase_scale = 1j * gam * (it_im_size - 1) / 2

        phase = np.exp(phase_scale * kern_in)
        full_coef.append(phase * coef)

        # nufft_offset
        koff = np.expand_dims(np.floor(it_om / gam - it_numpoints / 2), 1)
        Jvec = np.reshape(np.arange(1, it_numpoints + 1), (1, it_numpoints))
        kd.append(np.mod(Jvec + koff, it_grid_size) + 1)

    for i in range(len(kd)):
        kd[i] = (kd[i] - 1) * np.prod(grid_size[i + 1 :])

    # build the sparse matrix
    kk = kd[0]
    spmat_coef = full_coef[0]
    for i in range(1, ndims):
        Jprod = np.prod(numpoints[: i + 1])
        # block outer sum
        kk = np.reshape(
            np.expand_dims(kk, 1) + np.expand_dims(kd[i], 2), (klength, Jprod)
        )
        # block outer prod
        spmat_coef = np.reshape(
            np.expand_dims(spmat_coef, 1) * np.expand_dims(full_coef[i], 2),
            (klength, Jprod),
        )

    # build in fftshift
    phase = np.exp(1j * np.dot(np.transpose(omega), np.expand_dims(n_shift, 1)))
    spmat_coef = np.conj(spmat_coef) * phase

    # get coordinates in sparse matrix
    trajind = np.expand_dims(np.arange(klength), 1)
    trajind = np.repeat(trajind, np.prod(numpoints), axis=1)

    # build the sparse matrix
    spmat = coo_matrix(
        (spmat_coef.flatten(), (trajind.flatten(), kk.flatten())),
        shape=(klength, np.prod(grid_size)),
    )

    return spmat


def build_table(
    im_size: Sequence[int],
    grid_size: Sequence[int],
    numpoints: Sequence[int],
    table_oversamp: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> List[Tensor]:
    """Builds an interpolation table.

    Args:
        numpoints: Number of points to use for interpolation in each dimension.
        table_oversamp: Table oversampling factor.
        grid_size: Size of the grid to interpolate from.
        im_size: Size of base image.
        ndims: Number of image dimensions.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        A list of tables for each dimension.
    """
    table = []

    # build one table for each dimension
    for (
        it_im_size,
        it_grid_size,
        it_numpoints,
        it_table_oversamp,
        it_order,
        it_alpha,
    ) in zip(im_size, grid_size, numpoints, table_oversamp, order, alpha):
        # The following is a trick of Fessler.
        # It uses broadcasting semantics to quickly build the table.
        t1 = (
            it_numpoints / 2
            - 1
            + np.array(range(it_table_oversamp)) / it_table_oversamp
        )  # [L]
        om1 = t1 * 2 * np.pi / it_grid_size  # gam
        s1 = build_numpy_spmatrix(
            np.expand_dims(om1, 0),
            numpoints=(it_numpoints,),
            im_size=(it_im_size,),
            grid_size=(it_grid_size,),
            n_shift=(0,),
            order=(it_order,),
            alpha=(it_alpha,),
        )
        h = np.array(s1.getcol(it_numpoints - 1).todense())
        for col in range(it_numpoints - 2, -1, -1):
            h = np.concatenate((h, np.array(s1.getcol(col).todense())), axis=0)
        h = np.concatenate((h.flatten(), np.array([0])))

        table.append(torch.tensor(h))

    return table


def kaiser_bessel_ft(
    omega: np.ndarray, numpoints: int, alpha: float, order: float, d: int
) -> np.ndarray:
    """Computes FT of KB function for scaling in image domain.

    Args:
        om (ndarray): An array of coordinates to interpolate to.
        npts (int): Number of points to use for interpolation in each
            dimension.
        order (int): Order of Kaiser-Bessel kernel.
        alpha (double or array of doubles): KB parameter.
        d (int):  # TODO: find what d is

    Returns:
        The scaling coefficients.
    """
    z = np.sqrt((2 * np.pi * (numpoints / 2) * omega) ** 2 - alpha ** 2 + 0j)
    nu = d / 2 + order
    scaling_coef = (
        (2 * np.pi) ** (d / 2)
        * ((numpoints / 2) ** d)
        * (alpha ** order)
        / special.iv(order, alpha)
        * special.jv(nu, z)
        / (z ** nu)
    )
    scaling_coef = np.real(scaling_coef)

    return scaling_coef


def compute_scaling_coefs(
    im_size: Sequence[int],
    grid_size: Sequence[int],
    numpoints: Sequence[int],
    alpha: Sequence[float],
    order: Sequence[float],
) -> Tensor:
    """Computes scaling coefficients for NUFFT operation.

    Args:
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        numpoints: Number of points to use for interpolation in each dimension.
        alpha: KB parameter.
        order: Order of Kaiser-Bessel kernel.

    Returns:
        The scaling coefficients.
    """
    num_coefs = np.array(range(im_size[0])) - (im_size[0] - 1) / 2
    scaling_coef = 1 / kaiser_bessel_ft(
        num_coefs / grid_size[0], numpoints[0], alpha[0], order[0], 1
    )

    if numpoints[0] == 1:
        scaling_coef = np.ones(scaling_coef.shape)

    for i in range(1, len(im_size)):
        indlist = np.array(range(im_size[i])) - (im_size[i] - 1) / 2
        scaling_coef = np.expand_dims(scaling_coef, axis=-1)
        tmp = 1 / kaiser_bessel_ft(
            indlist / grid_size[i], numpoints[i], alpha[i], order[i], 1
        )

        for _ in range(i):
            tmp = tmp[np.newaxis]

        if numpoints[i] == 1:
            tmp = np.ones(tmp.shape)

        scaling_coef = scaling_coef * tmp

    return torch.tensor(scaling_coef)


def init_fn(
    im_size: Sequence[int],
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2 ** 10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
    dtype: torch.dtype = None,
):
    im_size = tuple(im_size)
    if grid_size is None:
        grid_size = tuple([dim * 2 for dim in im_size])
    else:
        grid_size = tuple(grid_size)
    if isinstance(numpoints, int):
        numpoints = tuple([numpoints for _ in range(len(grid_size))])
    else:
        numpoints = tuple(numpoints)
    if n_shift is None:
        n_shift = tuple([dim // 2 for dim in im_size])
    else:
        n_shift = tuple(n_shift)
    if isinstance(table_oversamp, int):
        table_oversamp = tuple(table_oversamp for _ in range(len(grid_size)))
    else:
        table_oversamp = tuple(table_oversamp)
    alpha = tuple(kbwidth * numpoint for numpoint in numpoints)
    if isinstance(order, float):
        order = tuple(order for _ in range(len(grid_size)))
    else:
        order = tuple(order)
    if dtype is None:
        dtype = torch.get_default_dtype()

    # dimension checking
    assert len(grid_size) == len(im_size)
    assert len(n_shift) == len(im_size)
    assert len(numpoints) == len(im_size)
    assert len(alpha) == len(im_size)
    assert len(order) == len(im_size)
    assert len(table_oversamp) == len(im_size)

    tables = build_table(
        numpoints=numpoints,
        table_oversamp=table_oversamp,
        grid_size=grid_size,
        im_size=im_size,
        order=order,
        alpha=alpha,
    )
    assert len(tables) == len(im_size)

    # precompute interpolation offsets
    offset_list = list(itertools.product(*[range(numpoint) for numpoint in numpoints]))

    if dtype.is_floating_point:
        real_dtype = dtype
        for pair in DTYPE_MAP:
            if pair[1] == real_dtype:
                complex_dtype = pair[0]
                break
    elif dtype.is_complex:
        complex_dtype = dtype
        for pair in DTYPE_MAP:
            if pair[0] == complex_dtype:
                real_dtype = pair[1]
                break
    else:
        raise TypeError("Unrecognized dtype.")

    tables = [table.to(complex_dtype) for table in tables]

    return (
        tables,
        torch.tensor(im_size, dtype=torch.long),
        torch.tensor(grid_size, dtype=torch.long),
        torch.tensor(n_shift, dtype=real_dtype),
        torch.tensor(numpoints, dtype=torch.long),
        torch.tensor(offset_list, dtype=torch.long),
        torch.tensor(table_oversamp, dtype=torch.long),
        torch.tensor(order, dtype=real_dtype),
        torch.tensor(alpha, dtype=real_dtype),
    )
