import pickle

import pytest
import torch
import torchkbnufft as tkbn

from .conftest import (
    create_input_plus_noise,
    create_ktraj,
)


def sense_nufft_adjoint_test(image, kdata, ktraj, smaps, forw_ob, adj_ob, spmat):
    image_forw = forw_ob(image, ktraj, smaps=smaps)
    kdata_adj = adj_ob(kdata, ktraj, smaps=smaps)

    assert torch.allclose(
        tkbn.inner_product(image_forw, kdata), tkbn.inner_product(image, kdata_adj)
    )

    image_forw = forw_ob(image, ktraj, spmat, smaps=smaps)
    kdata_adj = adj_ob(kdata, ktraj, spmat, smaps=smaps)

    assert torch.allclose(
        tkbn.inner_product(image_forw, kdata), tkbn.inner_product(image, kdata_adj)
    )


def sense_nufft_autograd_test(image, kdata, ktraj, smaps, forw_ob, adj_ob, spmat):
    image.requires_grad = True
    kdata.requires_grad = True
    image_forw = forw_ob(image, ktraj, smaps=smaps)
    kdata_adj = adj_ob(kdata, ktraj, smaps=smaps)

    (torch.abs(image_forw) ** 2 / 2).sum().backward()
    (torch.abs(kdata_adj) ** 2 / 2).sum().backward()
    autograd_forw = image.grad.clone()
    autograd_adj = kdata.grad.clone()
    grad_forw_est = adj_ob(image_forw.detach(), ktraj, smaps=smaps)
    grad_adj_est = forw_ob(kdata_adj.detach(), ktraj, smaps=smaps)

    assert torch.allclose(autograd_forw, grad_forw_est)
    assert torch.allclose(autograd_adj, grad_adj_est)

    image.grad = torch.zeros_like(image.grad)
    kdata.grad = torch.zeros_like(kdata.grad)
    image.requires_grad = True
    kdata.requires_grad = True
    image_forw = forw_ob(image, ktraj, spmat, smaps=smaps)
    kdata_adj = adj_ob(kdata, ktraj, spmat, smaps=smaps)

    (torch.abs(image_forw) ** 2 / 2).sum().backward()
    (torch.abs(kdata_adj) ** 2 / 2).sum().backward()
    autograd_forw = image.grad.clone()
    autograd_adj = kdata.grad.clone()
    grad_forw_est = adj_ob(image_forw.detach(), ktraj, spmat, smaps=smaps)
    grad_adj_est = forw_ob(kdata_adj.detach(), ktraj, spmat, smaps=smaps)

    assert torch.allclose(autograd_forw, grad_forw_est)
    assert torch.allclose(autograd_adj, grad_adj_est)


def test_sense_nufft_accuracy():
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    with open("tests/data/sense_nufft_data.pkl", "rb") as f:
        old_data = pickle.load(f)

    for (image, ktraj, smaps, old_kdata) in old_data:
        im_size = image.shape[2:-1]

        forw_ob = tkbn.KbNufft(im_size=im_size)

        kdata = forw_ob(image, ktraj, smaps=smaps)

        assert torch.allclose(kdata, old_kdata)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 3, 19], [1, 3, 25], True),
        ([3, 5, 13, 2], [3, 5, 18, 2], False),
        ([1, 4, 32, 16], [1, 4, 83], True),
        ([5, 8, 15, 12, 2], [5, 8, 83, 2], False),
        ([3, 10, 13, 18, 12], [3, 10, 112], True),
        ([1, 12, 17, 19, 12, 2], [1, 12, 112, 2], False),
    ],
)
def test_sense_nufft_adjoint(shape, kdata_shape, is_complex):
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
    kdata = create_input_plus_noise(kdata_shape, is_complex)
    smaps = create_input_plus_noise(shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size)

    # test with sparse matrices
    spmat = tkbn.build_tensor_spmatrix(
        ktraj,
        forw_ob.numpoints.numpy(),
        im_size,
        forw_ob.grid_size.tolist(),
        forw_ob.n_shift.numpy(),
        forw_ob.order.numpy(),
        forw_ob.alpha.numpy(),
    )

    sense_nufft_adjoint_test(image, kdata, ktraj, smaps, forw_ob, adj_ob, spmat)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 3, 19], [1, 3, 25], True),
        ([3, 5, 13, 2], [3, 5, 18, 2], False),
        ([1, 4, 32, 16], [1, 4, 83], True),
        ([5, 8, 15, 12, 2], [5, 8, 83, 2], False),
        ([3, 10, 13, 18, 12], [3, 10, 112], True),
        ([1, 12, 17, 19, 12, 2], [1, 12, 112, 2], False),
    ],
)
def test_sense_nufft_autograd(shape, kdata_shape, is_complex):
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
    kdata = create_input_plus_noise(kdata_shape, is_complex)
    smaps = create_input_plus_noise(shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size)

    # test with sparse matrices
    spmat = tkbn.build_tensor_spmatrix(
        ktraj,
        forw_ob.numpoints.numpy(),
        im_size,
        forw_ob.grid_size.tolist(),
        forw_ob.n_shift.numpy(),
        forw_ob.order.numpy(),
        forw_ob.alpha.numpy(),
    )

    sense_nufft_autograd_test(image, kdata, ktraj, smaps, forw_ob, adj_ob, spmat)

    torch.set_default_dtype(default_dtype)


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 3, 19], [1, 3, 25], True),
        ([1, 5, 32, 16], [1, 5, 83], True),
        ([3, 8, 13, 18, 12], [3, 8, 112], True),
    ],
)
def test_sense_nufft_complex_real_match(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    im_size = shape[2:]
    im_shape = [s for s in shape]
    im_shape[1] = 1

    image = create_input_plus_noise(im_shape, is_complex)
    smaps = create_input_plus_noise(shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbNufft(im_size=im_size)

    kdata_complex = forw_ob(image, ktraj, smaps=smaps)
    kdata_real = torch.view_as_complex(
        forw_ob(torch.view_as_real(image), ktraj, smaps=torch.view_as_real(smaps))
    )

    assert torch.allclose(kdata_complex, kdata_real)

    # test with sparse matrices
    spmat = tkbn.build_tensor_spmatrix(
        ktraj,
        forw_ob.numpoints.numpy(),
        im_size,
        forw_ob.grid_size.tolist(),
        forw_ob.n_shift.numpy(),
        forw_ob.order.numpy(),
        forw_ob.alpha.numpy(),
    )

    kdata_complex = forw_ob(image, ktraj, spmat, smaps=smaps)
    kdata_real = torch.view_as_complex(
        forw_ob(
            torch.view_as_real(image), ktraj, spmat, smaps=torch.view_as_real(smaps)
        )
    )

    assert torch.allclose(kdata_complex, kdata_real)

    torch.set_default_dtype(default_dtype)
