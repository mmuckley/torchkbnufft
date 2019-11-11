import torch

from ..math import complex_mult, conj_complex_mult
from ..functional.kbnufft import KbNufftFunction, AdjKbNufftFunction


def coilpack_sense_forward(x, smap, om, interpob, interp_mats=None):
    """Coil-packing SENSE-NUFFT operation.

    This function applies "coil-packing" SENSE-NUFFT operation, which is just a
    normal SENSE-NUFFT operation with a reshape command that puts the batch
    dimension into the coil dimension prior to the NUFFT. With the implementation
    in the package, the code will then broadcast NUFFT opeartions across the
    combined dimension, which is extremely fast.

    This is the fastest way to do NUFFT operations on a slice stack of multi-coil
    data, where the slices are stored in the batch dimension.

    Args:
        x (tensor): The input images of size (nbatch, ncoil, 2) + im_size.
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) + im_size.
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation. This is the fastest option, but more
            memory-intensive.
    Returns:
        y (tensor): Output off-grid k-space data of dimensions
            (nbatch, ncoil, 2, klength).
    """
    ncoil = smap.shape[1]

    # multiply coils
    x = complex_mult(x, smap, dim=2)

    # pack slice dim into coil dim
    new_sz = (1, -1, 2) + tuple(smap.shape[3:])
    x = x.view(*new_sz)

    # nufft
    y = KbNufftFunction.apply(x, om, interpob, interp_mats)

    # unpack slice dim from coil dim
    new_sz = (-1, ncoil, 2, y.shape[-1])
    y = y.view(*new_sz)

    return y


def coilpack_sense_backward(y, smap, om, interpob, interp_mats=None):
    """Coil-packing SENSE-NUFFT adjoint operation.

    This function applies "coil-packing" SENSE-NUFFT adjoint operation, which is
    a normal SENSE-NUFFT adjoint operation with a reshape command that puts the batch
    dimension into the coil dimension prior to the NUFFT. With the implementation
    in the package, the code will then broadcast NUFFT opeartions across the
    combined dimension, which is extremely fast.

    This is the fastest way to do NUFFT operations on a slice stack of multi-coil
    data, where the slices are stored in the batch dimension.

    Args:
        y (tensor): The input images of size (nbatch, ncoil, 2, klength).
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) + im_size.
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation. This is the fastest option, but more
            memory-intensive.
    Returns:
        x (tensor): The images after adjoint NUFFT of size
            (nbatch, ncoil, 2) + im_size.
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
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) + im_size.
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation. This is the fastest option, but more
            memory-intensive.
    Returns:
        y (tensor): Output off-grid k-space data of dimensions
            (nbatch, ncoil, 2, klength).
    """
    if isinstance(smap, torch.Tensor):
        dtype = smap.dtype
        device = smap.device
        mult_x = torch.zeros(smap.shape, dtype=dtype, device=device)
    else:
        mult_x = [None] * len(smap)

    # handle batch dimension
    for i, im in enumerate(x):
        # multiply sensitivities
        mult_x[i] = complex_mult(im, smap[i], dim=1)

    y = KbNufftFunction.apply(mult_x, om, interpob, interp_mats)

    return y


def sense_backward(y, smap, om, interpob, interp_mats=None):
    """SENSE-NUFFT adjoint operation.

    Args:
        y (tensor): The input images of size (nbatch, ncoil, 2, klength).
        smap (tensor): The sensitivity maps of size (nbatch, ncoil, 2) + im_size.
        interpob (dictionary): A NUFFT interpolation object.
        interp_mats (dictionary, default=None): A dictionary of sparse
            interpolation matrices. If not None, the NUFFT operation will use
            the matrices for interpolation. This is the fastest option, but more
            memory-intensive.
    Returns:
        x (tensor): The images after adjoint NUFFT of size
            (nbatch, ncoil, 2) + im_size.
    """
    # adjoint nufft
    x = AdjKbNufftFunction.apply(y, om, interpob, interp_mats)

    # conjugate sum
    out_x = []
    for i, im in enumerate(x):
        out_x.append(torch.sum(conj_complex_mult(
            im, smap[i], dim=1), dim=0, keepdim=True))

    if isinstance(smap, torch.Tensor):
        out_x = torch.stack(out_x)

    return out_x
