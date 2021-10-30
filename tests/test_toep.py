import numpy as np
import pytest
import torch
import torchkbnufft as tkbn

from .conftest import create_input_plus_noise, create_ktraj


@pytest.mark.parametrize(
    "shape, grid_size, kdata_shape, is_complex, norm",
    [
        ([1, 3, 19], [57], [1, 3, 25], True, "ortho"),
        ([3, 5, 13, 2], [19], [3, 5, 18, 2], False, None),
        ([1, 4, 32, 16], [64, 24], [1, 4, 83], True, None),
        ([5, 8, 15, 12, 2], [30, 24], [5, 8, 83, 2], False, "ortho"),
        ([3, 10, 13, 18, 12], [20, 26, 37], [3, 10, 112], True, None),
        ([1, 12, 17, 19, 12, 2], [25, 28, 24], [1, 12, 112, 2], False, "ortho"),
    ],
)
def test_toeplitz_nufft_accuracy(shape, grid_size, kdata_shape, is_complex, norm):
    norm_diff_tol = 1e-2  # toeplitz is only approximate
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]
    im_shape = [s for s in shape]
    im_shape[1] = 1

    image = create_input_plus_noise(im_shape, is_complex)
    smaps = create_input_plus_noise(shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)
    toep_ob = tkbn.ToepNufft()

    kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size, grid_size=grid_size, norm=norm)
    if not is_complex:
        kernel = torch.view_as_real(kernel)

    fbn = adj_ob(
        forw_ob(image, ktraj, smaps=smaps, norm=norm), ktraj, smaps=smaps, norm=norm,
    )
    fbt = toep_ob(image, kernel, smaps=smaps, norm=norm)

    if is_complex:
        fbn = torch.view_as_real(fbn)
        fbt = torch.view_as_real(fbt)

    norm_diff = torch.norm(fbn - fbt) / torch.norm(fbn)

    assert norm_diff < norm_diff_tol

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([4, 3, 19], [4, 3, 25], True),
        ([3, 5, 13, 2], [3, 5, 18, 2], False),
        ([2, 4, 32, 16], [2, 4, 83], True),
        ([5, 8, 15, 12, 2], [5, 8, 83, 2], False),
        ([3, 10, 13, 18, 12], [3, 10, 112], True),
        ([2, 12, 17, 19, 12, 2], [2, 12, 112, 2], False),
    ],
)
def test_batched_toeplitz_nufft_accuracy(shape, kdata_shape, is_complex):
    norm_diff_tol = 1e-4  # toeplitz is only approximate
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]
    im_shape = [s for s in shape]
    im_shape[1] = 1

    image = create_input_plus_noise(im_shape, is_complex)
    smaps = create_input_plus_noise(shape, is_complex)
    ktraj = (
        torch.rand(size=(shape[0], len(im_size), kdata_shape[2])) * 2 * np.pi - np.pi
    )

    forw_ob = tkbn.KbNufft(im_size=im_size)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size)
    toep_ob = tkbn.ToepNufft()

    kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size, norm="ortho")
    if not is_complex:
        kernel = torch.view_as_real(kernel)

    fbn = adj_ob(
        forw_ob(image, ktraj, smaps=smaps, norm="ortho"),
        ktraj,
        smaps=smaps,
        norm="ortho",
    )
    fbt = toep_ob(image, kernel, smaps=smaps, norm="ortho")

    if is_complex:
        fbn = torch.view_as_real(fbn)
        fbt = torch.view_as_real(fbt)

    norm_diff = torch.norm(fbn - fbt) / torch.norm(fbn)

    assert norm_diff < norm_diff_tol

    torch.set_default_dtype(default_dtype)
