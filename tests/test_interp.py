import pytest
import torch
import torchkbnufft as tkbn

from .conftest import create_input_plus_noise, create_ktraj


@pytest.mark.parametrize(
    "shape, kdata_shape, is_complex",
    [
        ([1, 1, 32, 16], [1, 1, 83], True),
        ([5, 1, 15, 12, 2], [5, 1, 83, 2], False),
        ([3, 2, 13, 18, 12], [3, 2, 112], True),
        ([1, 2, 17, 19, 12, 2], [1, 2, 112, 2], False),
    ],
)
def test_interp_adjoint(shape, kdata_shape, is_complex):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    image = create_input_plus_noise(shape, is_complex)
    kdata = create_input_plus_noise(kdata_shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    forw_ob = tkbn.KbInterpForward(im_size=im_size, grid_size=im_size)
    adj_ob = tkbn.KbInterpAdjoint(im_size=im_size, grid_size=im_size)

    image_forw = forw_ob(image, ktraj)
    kdata_adj = adj_ob(kdata, ktraj)

    assert torch.allclose(
        tkbn.inner_product(image_forw, kdata), tkbn.inner_product(image, kdata_adj)
    )

    # test with sparse matrices
    spmat = tkbn.build_tensor_spmatrix(
        ktraj,
        forw_ob.numpoints.numpy(),
        im_size,
        im_size,
        forw_ob.n_shift.numpy(),
        forw_ob.order.numpy(),
        forw_ob.alpha.numpy(),
    )

    image_forw = forw_ob(image, ktraj, spmat)
    kdata_adj = adj_ob(kdata, ktraj, spmat)

    assert torch.allclose(
        tkbn.inner_product(image_forw, kdata), tkbn.inner_product(image, kdata_adj)
    )
