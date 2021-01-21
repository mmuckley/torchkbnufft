import torch
from torch import Tensor
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore


def fft_old(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    return torch.fft(image, ndim, normalized)


def ifft_old(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    return torch.ifft(image, ndim, normalized)


def fft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))

    if image.is_complex():
        is_complex = True
    else:
        is_complex = False
        image = torch.view_as_complex(image)

    image = torch.fft.fftn(image, dim=dims, norm=norm)

    if not is_complex:
        image = torch.view_as_real(image)

    return image


def ifft_new(image: Tensor, ndim: int, normalized: bool = False) -> Tensor:
    norm = "ortho" if normalized else "forward"
    dims = tuple(range(-ndim, 0))

    if image.is_complex():
        is_complex = True
    else:
        is_complex = False
        image = torch.view_as_complex(image)

    image = torch.fft.ifftn(image, dim=dims, norm=norm)

    if not is_complex:
        image = torch.view_as_real(image)

    return image
