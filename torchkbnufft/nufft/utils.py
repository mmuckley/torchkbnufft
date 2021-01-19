from typing import Sequence, List

import numpy as np
import torch
from scipy import special
from scipy.sparse import coo_matrix
from torch import Tensor

from .fft_functions import ifft_and_scale_on_gridded_data, scale_and_fft_on_image_volume


def build_spmatrix(
    om: np.ndarray,
    numpoints: Sequence[int],
    im_size: Sequence[int],
    grid_size: Sequence[int],
    n_shift: Sequence[int],
    order: Sequence[float],
    alpha: Sequence[float],
) -> coo_matrix:
    """Builds a sparse matrix with the interpolation coefficients.

    Args:
        om: An array of coordinates to interpolate to.
        im_size: Size of base image.
        grid_size: Size of the grid to interpolate from.
        n_shift: Number of points to shift for fftshifts.
        order: Order of Kaiser-Bessel kernel.
        alpha: KB parameter.

    Returns:
        A scipy sparse interpolation matrix.
    """
    spmat = -1

    ndims = om.shape[0]
    klength = om.shape[1]

    # calculate interpolation coefficients using kb kernel
    def interp_coeff(om, npts, grdsz, alpha, order):
        gam = 2 * np.pi / grdsz
        interp_dist = om / gam - np.floor(om / gam - npts / 2)
        Jvec = np.reshape(np.array(range(1, npts + 1)), (1, npts))
        kern_in = -1 * Jvec + np.expand_dims(interp_dist, 1)

        cur_coeff = np.zeros(shape=kern_in.shape, dtype=np.complex)
        indices = abs(kern_in) < npts / 2
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
    ) in zip(om, im_size, grid_size, numpoints, om, alpha, order):
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
        Jvec = np.reshape(np.array(range(1, it_numpoints + 1)), (1, it_numpoints))
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
    phase = np.exp(1j * np.dot(np.transpose(om), np.expand_dims(n_shift, 1)))
    spmat_coef = np.conj(spmat_coef) * phase

    # get coordinates in sparse matrix
    trajind = np.expand_dims(np.array(range(klength)), 1)
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
        s1 = build_spmatrix(
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


def kaiser_bessel_ft(om, npts, alpha, order, d):
    """Computes FT of KB function for scaling in image domain.

    Args:
        om (ndarray): An array of coordinates to interpolate to.
        npts (int): Number of points to use for interpolation in each
            dimension.
        order (int): Order of Kaiser-Bessel kernel.
        alpha (double or array of doubles): KB parameter.
        d (int):  # TODO: find what d is

    Returns:
        ndarray: The scaling coefficients.
    """
    z = np.sqrt((2 * np.pi * (npts / 2) * om) ** 2 - alpha ** 2 + 0j)
    nu = d / 2 + order
    scaling_coef = (
        (2 * np.pi) ** (d / 2)
        * ((npts / 2) ** d)
        * (alpha ** order)
        / special.iv(order, alpha)
        * special.jv(nu, z)
        / (z ** nu)
    )
    scaling_coef = np.real(scaling_coef)

    return scaling_coef


def compute_scaling_coefs(im_size, grid_size, numpoints, alpha, order):
    """Computes scaling coefficients for NUFFT operation.

    Args:
        im_size (tuple): Size of base image.
        grid_size (tuple): Size of the grid to interpolate from.
        numpoints (tuple): Number of points to use for interpolation in each
            dimension. Default is six points in each direction.
        alpha (tuple): KB parameter.
        order (tuple): Order of Kaiser-Bessel kernel.

    Returns:
        ndarray: The scaling coefficients.
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

    return scaling_coef
