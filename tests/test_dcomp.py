import pytest
import torch
import torchkbnufft as tkbn

from .conftest import create_input_plus_noise, create_ktraj


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
def test_dcomp_run(shape, kdata_shape, is_complex):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    torch.manual_seed(123)
    if is_complex:
        im_size = shape[2:]
    else:
        im_size = shape[2:-1]

    kdata = create_input_plus_noise(kdata_shape, is_complex)
    ktraj = create_ktraj(len(im_size), kdata_shape[2])

    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size)
    dcomp = tkbn.calc_density_compensation_function(ktraj=ktraj, im_size=im_size)

    if not is_complex:
        dcomp = torch.view_as_real(dcomp)

    _ = adj_ob(kdata * dcomp, ktraj)

    torch.set_default_dtype(default_dtype)
