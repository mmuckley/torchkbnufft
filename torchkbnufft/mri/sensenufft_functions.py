import torch

from ..functional.kbnufft import AdjKbNufftFunction, KbNufftFunction
from ..math import complex_mult, conj_complex_mult
from ..nufft.fft_functions import fft_filter


def coilpack_sense_forward(x, smap, om, interpob, interp_mats=None):
    """Coil-packing SENSE-NUFFT operation.

    This function applies "coil-packing" SENSE-NUFFT operation, which is just a
    normal SENSE-NUFFT operation with a reshape command that puts the batch
    dimension into the coil dimension prior to the NUFFT. With the
    implementation in the package, the code will then broadcast NUFFT
    operations across the combined dimension, which is extremely fast.

    This is the fastest way to do NUFFT operations on a slice stack of
    multi-coil data, where the slices are stored in the batch dimension.

    Args:
        x (tensor): The input images of size (nbatch, ncoil, 2) + im_size.
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) +
            im_size.
        om (tensor): The k-space trajectory in units of radians/voxel of size
            (1, ndims, klength).
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation.

    Returns:
        tensor: Output off-grid k-space data of dimensions (nbatch, ncoil, 2,
            klength).
    """
    ncoil = smap.shape[1]

    # multiply coils
    y = complex_mult(x, smap, dim=2)

    # pack slice dim into coil dim
    new_sz = (1, -1, 2) + tuple(smap.shape[3:])
    y = y.view(*new_sz)

    # nufft
    y = KbNufftFunction.apply(y, om, interpob, interp_mats)

    # unpack slice dim from coil dim
    new_sz = (-1, ncoil, 2, y.shape[-1])
    y = y.view(*new_sz)

    return y


def coilpack_sense_backward(y, smap, om, interpob, interp_mats=None):
    """Coil-packing SENSE-NUFFT adjoint operation.

    This function applies "coil-packing" SENSE-NUFFT adjoint operation, which is
    a normal SENSE-NUFFT adjoint operation with a reshape command that puts the
    batch dimension into the coil dimension prior to the NUFFT. With the
    implementation in the package, the code will then broadcast NUFFT operations
    across the combined dimension, which is extremely fast.

    This is the fastest way to do NUFFT operations on a slice stack of
    multi-coil data, where the slices are stored in the batch dimension.

    Args:
        y (tensor): The input images of size (nbatch, ncoil, 2, klength).
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) +
            im_size.
        om (tensor): The k-space trajectory in units of radians/voxel of size
            (1, ndims, klength).
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation.

    Returns:
        tensor: The images after adjoint NUFFT of size (nbatch, ncoil, 2) +
            im_size.
    """
    ncoil = smap.shape[1]

    # pack slice dim into coil dim
    new_sz = (1, -1, 2, y.shape[-1])
    y = y.view(*new_sz)

    # adjoint nufft
    x = AdjKbNufftFunction.apply(y, om, interpob, interp_mats)

    # unpack slice dim from coil dim
    new_sz = (-1, ncoil, 2) + tuple(smap.shape[3:])
    x = x.view(*new_sz)

    # conjugate sum
    x = torch.sum(conj_complex_mult(x, smap, dim=2), dim=1, keepdim=True)

    return x


def sense_forward(x, smap, om, interpob, interp_mats=None):
    """SENSE-NUFFT operation.

    Args:
        x (tensor): The input images of size (nbatch, ncoil, 2) + im_size.
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) +
            im_size.
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation.

    Returns:
        tensor: Output off-grid k-space data of dimensions (nbatch, ncoil, 2,
            klength).
    """
    if isinstance(smap, torch.Tensor):
        dtype = smap.dtype
        device = smap.device
        y = torch.zeros(smap.shape, dtype=dtype, device=device)
    else:
        y = [None] * len(smap)

    # handle batch dimension
    for i, im in enumerate(x):
        # multiply sensitivities
        y[i] = complex_mult(im, smap[i], dim=1)

    y = KbNufftFunction.apply(y, om, interpob, interp_mats)

    return y


def sense_backward(y, smap, om, interpob, interp_mats=None):
    """SENSE-NUFFT adjoint operation.

    Args:
        y (tensor): The input images of size (nbatch, ncoil, 2, klength).
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) +
            im_size.
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation.

    Returns:
        tensor: The images after adjoint NUFFT of size (nbatch, ncoil, 2) +
            im_size.
    """
    # adjoint nufft
    x = AdjKbNufftFunction.apply(y, om, interpob, interp_mats)

    # conjugate sum
    x = list(x)
    for i in range(len(x)):
        x[i] = torch.sum(conj_complex_mult(
            x[i], smap[i], dim=1), dim=0, keepdim=True)

    if isinstance(smap, torch.Tensor):
        x = torch.stack(x)

    return x


def sense_toeplitz(x, smap, kern, norm=None):
    """Forward/Adjoint SENSE-NUFFT with Toeplitz embedding.

    This function applies both a forward and adjoint SENSE-NUFFT with Toeplitz
    embedding for the NUFFT operations, thus avoiding any gridding or
    interpolation and using only FFTs (very fast).

    Args:
        x (tensor): The input images of size (nbatch, 1, 2) + im_size.
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) +
            im_size.
        kern (tensor): Embedded Toeplitz NUFFT kernel of size
            (nbatch, ncoil, 2) + im_size*2.
        norm (str, default=None): If 'ortho', use orthogonal FFTs for Toeplitz
            NUFFT filter.

    Returns:
        tensor: The images after forward and adjoint NUFFT of size
            (nbatch, 1, 2) + im_size.
    """
    x = list(x)

    # handle batch dimension to avoid exploding memory
    for i in range(len(x)):
        x[i] = _sense_toep_filt(x[i], smap[i], kern[i], norm)

    x = torch.stack(x)

    return x


def _sense_toep_filt(x, smap, kern, norm):
    """Subroutine for sense_toeplitz().

    Args:
        x (tensor): The input images of size (1, 2) + im_size.
        smap (tensor): The sensitivity maps of size (ncoil, 2) +
            im_size.
        kern (tensor): Embedded Toeplitz NUFFT kernel of size
            (ncoil, 2) + im_size*2.
        norm (str, default=None): If 'ortho', use orthogonal FFTs for Toeplitz
            NUFFT filter.

    Returns:
        tensor: The images after forward and adjoint NUFFT of size
            (1, 2) + im_size.
    """
    # multiply sensitivities
    x = complex_mult(x, smap, dim=1)

    # Toeplitz NUFFT
    x = fft_filter(
        x.unsqueeze(0),
        kern.unsqueeze(0),
        norm=norm
    ).squeeze(0)

    # conjugate sum
    x = torch.sum(conj_complex_mult(x, smap, dim=1), dim=0, keepdim=True)

    return x
