import pickle

import numpy as np
import pytest
import torch
import torchkbnufft as tkbn

from .conftest import (
    create_input_plus_noise,
    create_ktraj,
    nufft_adjoint_test,
    nufft_autograd_test,
)


def test_interp_accuracy():
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    with open("tests/data/interp_data.pkl", "rb") as f:
        old_data = pickle.load(f)

    for (image, ktraj, old_kdata) in old_data:
        im_size = image.shape[2:-1]

        forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size)

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
def test_interp_adjoint(shape, kdata_shape, is_complex):
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

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
        grid_size=im_size,
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
def test_interp_autograd(shape, kdata_shape, is_complex):
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

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
        grid_size=im_size,
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
def test_interp_complex_real_match(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    im_size = shape[2:]

    image = create_input_plus_noise(shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size)

    kdata_complex = forw_ob(image, ktraj)
    kdata_real = torch.view_as_complex(forw_ob(torch.view_as_real(image), ktraj))

    assert torch.allclose(kdata_complex, kdata_real)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
        grid_size=im_size,
    )

    kdata_complex = forw_ob(image, ktraj, spmat)
    kdata_real = torch.view_as_complex(forw_ob(torch.view_as_real(image), ktraj, spmat))

    assert torch.allclose(kdata_complex, kdata_real)

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
def test_interp_cpu_gpu_match(shape, kdata_shape, is_complex):
    if not torch.cuda.is_available():
        pytest.skip()

    device = torch.device("cpu")

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex).to(device)
    ktraj = create_ktraj(len(im_size), kdata_shape[2]).to(device)

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size).to(device)

    kdat_cpu = forw_ob(image, ktraj)

    device = torch.device("cuda")
    image = image.to(device)
    ktraj = ktraj.to(device)
    forw_ob = forw_ob.to(device)

    kdat_gpu = forw_ob(image, ktraj).cpu()

    assert torch.allclose(kdat_cpu, kdat_gpu)

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
def test_interp_adjoint_gpu(shape, kdata_shape, is_complex):
    if not torch.cuda.is_available():
        pytest.skip()

    device = torch.device("cuda")

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex).to(device)
    kdata = create_input_plus_noise(kdata_shape, is_complex).to(device)
    ktraj = create_ktraj(len(im_size), kdata_shape[2]).to(device)

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size).to(device)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size).to(device)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
        grid_size=im_size,
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
def test_interp_autograd_gpu(shape, kdata_shape, is_complex):
    if not torch.cuda.is_available():
        pytest.skip()

    device = torch.device("cuda")

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex).to(device)
    kdata = create_input_plus_noise(kdata_shape, is_complex).to(device)
    ktraj = create_ktraj(len(im_size), kdata_shape[2]).to(device)

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size).to(device)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size).to(device)

    # test with sparse matrices
    spmat = tkbn.calc_tensor_spmatrix(
        ktraj,
        im_size,
        grid_size=im_size,
    )

    nufft_autograd_test(image, kdata, ktraj, forw_ob, adj_ob, spmat)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([3, 1, 19], [3, 1, 25], True),
        ([3, 1, 13, 2], [3, 1, 18, 2], False),
        ([4, 1, 32, 16], [4, 1, 83], True),
        ([5, 1, 15, 12, 2], [5, 1, 83, 2], False),
        ([3, 2, 13, 18, 12], [3, 2, 112], True),
        ([2, 2, 17, 19, 12, 2], [2, 2, 112, 2], False),
    ],
)
def test_interp_batches(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex)
    kdata = create_input_plus_noise(kdata_shape, is_complex)
    ktraj = (
        torch.rand(size=(shape[0], len(im_size), kdata_shape[2])) * 2 * np.pi - np.pi
    )

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size)

    forloop_test_forw = []
    for image_it, ktraj_it in zip(image, ktraj):
        forloop_test_forw.append(forw_ob(image_it.unsqueeze(0), ktraj_it))

    batched_test_forw = forw_ob(image, ktraj)

    assert torch.allclose(torch.cat(forloop_test_forw), batched_test_forw)

    forloop_test_adj = []
    for data_it, ktraj_it in zip(kdata, ktraj):
        forloop_test_adj.append(adj_ob(data_it.unsqueeze(0), ktraj_it))

    batched_test_adj = adj_ob(kdata, ktraj)

    assert torch.allclose(torch.cat(forloop_test_adj), batched_test_adj)

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

    forw_ob = tkbn.KbInterp(im_size=im_size, grid_size=im_size).to(image)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size).to(kdata)

    _ = forw_ob(image, ktraj)
    _ = adj_ob(kdata, ktraj)
