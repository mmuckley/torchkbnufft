import pickle

import pytest
import torch
import torchkbnufft as tkbn

from .conftest import (
    create_input_plus_noise,
    create_ktraj,
    nufft_adjoint_test,
    nufft_autograd_test,
)


def test_nufft_accuracy():
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    with open("tests/data/nufft_data.pkl", "rb") as f:
        old_data = pickle.load(f)

    for (image, ktraj, old_kdata) in old_data:
        im_size = image.shape[2:-1]

        forw_ob = tkbn.KbNufft(im_size=im_size)

        kdata = forw_ob(image, ktraj)

        assert torch.allclose(kdata, old_kdata)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 1, 19], [1, 1, 25], True),
        ([3, 1, 13, 2], [3, 1, 18, 2], False),
        ([1, 1, 32, 16], [1, 1, 83], True),
        ([5, 1, 15, 12, 2], [5, 1, 83, 2], False),
        ([3, 2, 13, 18, 12], [3, 2, 112], True),
        ([1, 2, 17, 19, 12, 2], [1, 2, 112, 2], False),
    ],
)
def test_nufft_adjoint(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex)
    kdata = create_input_plus_noise(kdata_shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
    )

    nufft_adjoint_test(image, kdata, ktraj, forw_ob, adj_ob, spmat)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 1, 19], [1, 1, 25], True),
        ([3, 1, 13, 2], [3, 1, 18, 2], False),
        ([1, 1, 32, 16], [1, 1, 83], True),
        ([5, 1, 15, 12, 2], [5, 1, 83, 2], False),
        ([3, 2, 13, 18, 12], [3, 2, 112], True),
        ([1, 2, 17, 19, 12, 2], [1, 2, 112, 2], False),
    ],
)
def test_nufft_autograd(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex)
    kdata = create_input_plus_noise(kdata_shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
    )

    nufft_autograd_test(image, kdata, ktraj, forw_ob, adj_ob, spmat)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 1, 19], [1, 1, 25], True),
        ([1, 1, 32, 16], [1, 1, 83], True),
        ([3, 2, 13, 18, 12], [3, 2, 112], True),
    ],
)
def test_nufft_complex_real_match(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    im_size = shape[2:]

    image = create_input_plus_noise(shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size)

    kdata_complex = forw_ob(image, ktraj)
    kdata_real = torch.view_as_complex(forw_ob(torch.view_as_real(image), ktraj))

    assert torch.allclose(kdata_complex, kdata_real)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
    )

    kdata_complex = forw_ob(image, ktraj, spmat)
    kdata_real = torch.view_as_complex(forw_ob(torch.view_as_real(image), ktraj, spmat))

    assert torch.allclose(kdata_complex, kdata_real)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex, dtype",
    [
        ([1, 1, 19], [1, 1, 25], True, torch.complex64),
        ([3, 1, 13, 2], [3, 1, 18, 2], False, torch.float32),
        ([1, 1, 19], [1, 1, 25], True, torch.complex128),
        ([3, 1, 13, 2], [3, 1, 18, 2], False, torch.float64),
    ],
)
def test_nufft_dtype_transfer(shape, kdata_shape, is_complex, dtype):
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    if dtype == torch.complex64:
        real_dtype = torch.float32
    elif dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        real_dtype = dtype

    image = create_input_plus_noise(shape, is_complex).to(dtype)
    kdata = create_input_plus_noise(kdata_shape, is_complex).to(dtype)
    ktraj = create_ktraj(len(im_size), kdata_shape[2]).to(real_dtype)

    forw_ob = tkbn.KbNufft(im_size=im_size).to(image)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size).to(kdata)

    _ = forw_ob(image, ktraj)
    _ = adj_ob(kdata, ktraj)
