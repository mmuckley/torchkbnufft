import numpy as np
import torch
import torch.nn.functional as F

from ..math import complex_mult, conj_complex_mult


def scale_and_fft_on_image_volume(x, scaling_coef, grid_size, im_size, norm):
    """Applies the FFT and any relevant scaling factors to x.

    Args:
        x (tensor): The image to be FFT'd.
        scaling_coef (tensor): The NUFFT scaling coefficients to be multiplied
            prior to FFT.
        grid_size (tensor): The oversampled grid size.
        im_size (tensor): The image dimensions for x.
        norm (str): Type of normalization factor to use. If 'ortho', uses
            orthogonal FFT, otherwise, no normalization is applied.

    Returns:
        tensor: The oversampled FFT of x.
    """
    # zero pad for oversampled nufft
    # also calculate fft permutations
    pad_sizes = []
    permute_dims = [0, 1]
    inv_permute_dims = [0, 1, 2 + grid_size.shape[0]]
    for i in range(grid_size.shape[0]):
        pad_sizes.append(0)
        pad_sizes.append(int(grid_size[-1 - i] - im_size[-1 - i]))
        permute_dims.append(3 + i)
        inv_permute_dims.append(2 + i)
    permute_dims.append(2)
    pad_sizes = tuple(pad_sizes)
    permute_dims = tuple(permute_dims)
    inv_permute_dims = tuple(inv_permute_dims)
    while len(scaling_coef.shape) < len(x.shape):
        scaling_coef = scaling_coef.unsqueeze(0)

    # multiply by scaling coefs
    x = complex_mult(x, scaling_coef, dim=2)

    # zero pad and fft
    x = F.pad(x, pad_sizes)
    x = x.permute(permute_dims)
    x = torch.fft(x, grid_size.numel())
    if norm == 'ortho':
        x = x / torch.sqrt(torch.prod(grid_size))
    x = x.permute(inv_permute_dims)

    return x


def ifft_and_scale_on_gridded_data(x, scaling_coef, grid_size, im_size, norm):
    """Applies the iFFT and any relevant scaling factors to x.

    Args:
        x (tensor): The image to be iFFT'd.
        scaling_coef (tensor): The NUFFT scaling coefficients to be multiplied
            after iFFT.
        grid_size (tensor): The oversampled grid size.
        im_size (tensor): The image dimensions for x.
        norm (str): Type of normalization factor to use. If 'ortho', uses
            orthogonal iFFT, otherwise, no normalization is applied.

    Returns:
        tensor: The iFFT of x.
    """
    # permutations for torch fft
    permute_dims = [0, 1]
    inv_permute_dims = [0, 1, 2 + grid_size.shape[0]]
    for i in range(grid_size.shape[0]):
        permute_dims.append(3 + i)
        inv_permute_dims.append(2 + i)
    permute_dims.append(2)
    permute_dims = tuple(permute_dims)
    inv_permute_dims = tuple(inv_permute_dims)

    # do the inverse fft
    x = x.permute(permute_dims)
    x = torch.ifft(x, grid_size.numel())
    x = x.permute(inv_permute_dims)

    # crop to output size
    crop_starts = tuple(np.array(x.shape).astype(np.int) * 0)
    crop_ends = [x.shape[0], x.shape[1], x.shape[2]]
    for dim in im_size:
        crop_ends.append(int(dim))
    x = x[tuple(map(slice, crop_starts, crop_ends))]

    # scaling
    if norm == 'ortho':
        x = x * torch.sqrt(torch.prod(grid_size))
    else:
        x = x * torch.prod(grid_size)

    # scaling coefficient multiply
    while len(scaling_coef.shape) < len(x.shape):
        scaling_coef = scaling_coef.unsqueeze(0)

    # try to broadcast multiply - batch over coil if not enough memory
    raise_error = False
    try:
        x = conj_complex_mult(x, scaling_coef, dim=2)
    except RuntimeError as e:
        if 'out of memory' in str(e) and not raise_error:
            torch.cuda.empty_cache()
            for coilind in range(x.shape[1]):
                x[:, coilind, ...] = conj_complex_mult(
                    x[:, coilind:coilind + 1, ...], scaling_coef, dim=2)
            raise_error = True
        else:
            raise e
    except BaseException:
        raise e

    return x


def fft_filter(x, kern, norm=None):
    """FFT-based filtering on a 2-size oversampled grid.
    """
    im_size = torch.tensor(x.shape).to(torch.long)[3:]
    grid_size = im_size * 2

    # set up n-dimensional zero pad
    pad_sizes = []
    permute_dims = [0, 1]
    inv_permute_dims = [0, 1, 2 + grid_size.shape[0]]
    for i in range(grid_size.shape[0]):
        pad_sizes.append(0)
        pad_sizes.append(int(grid_size[-1 - i] - im_size[-1 - i]))
        permute_dims.append(3 + i)
        inv_permute_dims.append(2 + i)
    permute_dims.append(2)
    pad_sizes = tuple(pad_sizes)
    permute_dims = tuple(permute_dims)
    inv_permute_dims = tuple(inv_permute_dims)

    # zero pad and fft
    x = F.pad(x, pad_sizes)
    x = x.permute(permute_dims)
    x = torch.fft(x, grid_size.numel())
    if norm == 'ortho':
        x = x / torch.sqrt(torch.prod(grid_size.to(torch.double)))
    x = x.permute(inv_permute_dims)

    # apply the filter
    x = complex_mult(x, kern, dim=2)

    # inverse fft
    x = x.permute(permute_dims)
    x = torch.ifft(x, grid_size.numel())
    x = x.permute(inv_permute_dims)

    # crop to input size
    crop_starts = tuple(np.array(x.shape).astype(np.int) * 0)
    crop_ends = [x.shape[0], x.shape[1], x.shape[2]]
    for dim in im_size:
        crop_ends.append(int(dim))
    x = x[tuple(map(slice, crop_starts, crop_ends))]

    # scaling, assume user handled adjoint scaling with their kernel
    if norm == 'ortho':
        x = x / torch.sqrt(torch.prod(grid_size.to(torch.double)))

    return x
