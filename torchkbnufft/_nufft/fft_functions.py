from typing import Optional

import torch
import torch.nn.functional as F
from packaging import version
from torch import Tensor

from .._math import complex_mult, conj_complex_mult

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    from .fft_compatibility import fft_new as fft_fn
    from .fft_compatibility import ifft_new as ifft_fn
else:
    from .fft_compatibility import fft_old as fft_fn
    from .fft_compatibility import ifft_old as ifft_fn


def crop_dims(image: Tensor, dim_list: Tensor, end_list: Tensor):
    if image.is_complex():
        is_complex = True
        image = torch.view_as_real(image)  # index select only works for real
    else:
        is_complex = False

    for (dim, end) in zip(dim_list, end_list):
        image = torch.index_select(image, dim, torch.arange(end))

    if is_complex:
        image = torch.view_as_complex(image)

    return image


def fft_and_scale(
    image: Tensor,
    scaling_coef: Tensor,
    grid_size: Tensor,
    im_size: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    """Applies the FFT and any relevant scaling factors to x.

    Args:
        image: The image to be FFT'd.
        scaling_coef: The NUFFT scaling coefficients to be multiplied prior to
            FFT.
        grid_size: The oversampled grid size.
        im_size: The image dimensions for x.
        norm; Optional: Type of normalization factor to use. If 'ortho', uses
            orthogonal FFT, otherwise, no normalization is applied.

    Returns:
        The oversampled FFT of image.
    """
    if norm is not None and norm != "ortho":
        raise ValueError("Only option for norm is 'ortho'.")

    normalized = True if norm == "ortho" else False

    # zero pad for oversampled nufft
    if image.is_complex():
        pad_sizes = []
    else:
        pad_sizes = [0, 0]
    for (gd, im) in zip(grid_size.flip((0,)), im_size.flip((0,))):
        pad_sizes.append(0)
        pad_sizes.append(int(gd - im))

    # scaling coef unsqueeze
    if not image.is_complex():
        scaling_coef = torch.view_as_real(scaling_coef)

    scaling_coef = scaling_coef.unsqueeze(0).unsqueeze(0)

    # multiply by scaling_coef, pad, then fft
    return fft_fn(
        F.pad(complex_mult(image, scaling_coef), tuple(pad_sizes)),
        grid_size.numel(),
        normalized=normalized,
    )


def ifft_and_scale(
    image: Tensor,
    scaling_coef: Tensor,
    grid_size: Tensor,
    im_size: Tensor,
    norm: Optional[str] = None,
):
    """Applies the iFFT and any relevant scaling factors to x.

    Args:
        image: The image to be iFFT'd.
        scaling_coef: The NUFFT scaling coefficients to be multiplied after
            iFFT.
        grid_size: The oversampled grid size.
        im_size: The image dimensions for x.
        norm: Type of normalization factor to use. If 'ortho', uses orthogonal
            iFFT, otherwise, no normalization is applied.

    Returns:
        The iFFT of image.
    """
    if norm is not None and norm != "ortho":
        raise ValueError("Only option for norm is 'ortho'.")

    normalized = True if norm == "ortho" else False

    # calculate crops
    dims = torch.arange(len(im_size)) + 2

    # scaling coef unsqueeze
    if not image.is_complex():
        scaling_coef = torch.view_as_real(scaling_coef)

    scaling_coef = scaling_coef.unsqueeze(0).unsqueeze(0)

    # ifft, crop, then multiply by scaling_coef conjugate
    return conj_complex_mult(
        crop_dims(
            ifft_fn(image, grid_size.numel(), normalized=normalized), dims, im_size
        ),
        scaling_coef,
    )


def fft_filter(image: Tensor, kernel: Tensor, norm: Optional[str] = None):
    """FFT-based filtering on a 2-size oversampled grid."""
    if norm is not None and norm != "ortho":
        raise ValueError("Only option for norm is 'ortho'.")

    normalized = True if norm == "ortho" else False

    if image.is_complex:
        im_size = torch.tensor(image.shape[2:], dtype=torch.long, device=image.device)
    else:
        im_size = torch.tensor(image.shape[2:-1], dtype=torch.long, device=image.device)

    grid_size = im_size * 2

    # set up n-dimensional zero pad
    if image.is_complex():
        pad_sizes = []
    else:
        pad_sizes = [0, 0]
    for (gd, im) in zip(grid_size.flip((0,)), im_size.flip((0,))):
        pad_sizes.append(0)
        pad_sizes.append(int(gd - im))

    # calculate crops
    dims = torch.arange(len(im_size)) + 2

    # pad, forward fft, multiply filter kernel, inverse fft, then crop pad
    return crop_dims(
        ifft_fn(
            complex_mult(
                fft_fn(
                    F.pad(image, pad_sizes), grid_size.numel(), normalized=normalized
                ),
                kernel,
            ),
            grid_size.numel(),
            normalized=normalized,
        ),
        dims,
        im_size,
    )
