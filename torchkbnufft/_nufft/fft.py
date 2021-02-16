from typing import List, Optional

import torch
import torch.fft
import torch.nn.functional as F
from torch import Tensor


def fft_fn(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    """Function for managing FFT normalizations."""
    norm = "ortho" if normalized else None
    dims = [el for el in range(-ndim, 0)]

    return torch.fft.fftn(image, dim=dims, norm=norm)  # type: ignore


def ifft_fn(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    """Function for managing FFT normalizations."""
    norm = "ortho" if normalized else "forward"
    dims = [el for el in range(-ndim, 0)]

    return torch.fft.ifftn(image, dim=dims, norm=norm)  # type: ignore


def crop_dims(image: Tensor, dim_list: Tensor, end_list: Tensor) -> Tensor:
    """Crops an n-dimensional Tensor."""
    image = torch.view_as_real(image)  # index select only works for real

    for (dim, end) in zip(dim_list, end_list):
        image = torch.index_select(image, dim, torch.arange(end, device=image.device))

    return torch.view_as_complex(image)


@torch.jit.script
def fft_and_scale(
    image: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    """Applies the FFT and any relevant scaling factors.

    Args:
        image: The image to be FFT'd.
        scaling_coef: The NUFFT scaling coefficients to be multiplied prior to
            FFT.
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        norm; Optional: Type of normalization factor to use. If 'ortho', uses
            orthogonal FFT, otherwise, no normalization is applied.

    Returns:
        The oversampled FFT of image.
    """
    normalized = False
    if norm is not None:
        if norm == "ortho":
            normalized = True
        else:
            raise ValueError("Only option for norm is 'ortho'.")

    # zero pad for oversampled nufft
    pad_sizes: List[int] = []
    for (gd, im) in zip(grid_size.flip((0,)), im_size.flip((0,))):
        pad_sizes.append(0)
        pad_sizes.append(int(gd - im))

    # multiply by scaling_coef, pad, then fft
    return fft_fn(
        F.pad(image * scaling_coef, pad_sizes),
        grid_size.numel(),
        normalized=normalized,
    )


@torch.jit.script
def ifft_and_scale(
    image: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    """Applies the iFFT and any relevant scaling factors.

    Args:
        image: The image to be iFFT'd.
        scaling_coef: The NUFFT scaling coefficients to be conjugate multiplied
            after to FFT.
        im_size: Size of image.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times `im_size`.
        norm; Optional: Type of normalization factor to use. If 'ortho', uses
            orthogonal FFT, otherwise, no normalization is applied.

    Returns:
        The iFFT of `image`.
    """
    normalized = False
    if norm is not None:
        if norm == "ortho":
            normalized = True
        else:
            raise ValueError("Only option for norm is 'ortho'.")

    # calculate crops
    dims = torch.arange(len(im_size), device=image.device) + 2

    # ifft, crop, then multiply by scaling_coef conjugate
    return (
        crop_dims(
            ifft_fn(image, grid_size.numel(), normalized=normalized), dims, im_size
        )
        * scaling_coef.conj()
    )


def fft_filter(image: Tensor, kernel: Tensor, norm: Optional[str] = "ortho") -> Tensor:
    r"""FFT-based filtering on an oversampled grid.

    This is a wrapper for the operation

    .. math::
        \text{output} = iFFT(\text{kernel}*FFT(\text{image}))

    where :math:`iFFT` and :math:`FFT` are both implemented as oversampled
    FFTs.

    Args:
        image: The image to be filtered.
        kernel: FFT-domain filter.
        norm: Whether to apply normalization with the FFT operation. Options
            are ``"ortho"`` or ``None``.

    Returns:
        Filtered version of ``image``.
    """
    normalized = False
    if norm is not None:
        if norm == "ortho":
            normalized = True
        else:
            raise ValueError("Only option for norm is 'ortho'.")

    im_size = torch.tensor(image.shape[2:], dtype=torch.long, device=image.device)
    grid_size = torch.tensor(
        kernel.shape[-len(image.shape[2:]) :], dtype=torch.long, device=image.device
    )

    # set up n-dimensional zero pad
    # zero pad for oversampled nufft
    pad_sizes: List[int] = []
    for (gd, im) in zip(grid_size.flip((0,)), im_size.flip((0,))):
        pad_sizes.append(0)
        pad_sizes.append(int(gd - im))

    # calculate crops
    dims = torch.arange(len(im_size), device=image.device) + 2

    # pad, forward fft, multiply filter kernel, inverse fft, then crop pad
    return crop_dims(
        ifft_fn(
            fft_fn(F.pad(image, pad_sizes), grid_size.numel(), normalized=normalized)
            * kernel,
            grid_size.numel(),
            normalized=normalized,
        ),
        dims,
        im_size,
    )
