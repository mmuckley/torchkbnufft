import torch
import numpy as np


def test_params():
    batch_size = 5
    ncoil = 4
    klength = 112

    im_size_2d = (33, 24)
    im_size_3d = (11, 33, 24)

    numpoints_2d = (4, 6)
    numpoints_3d = (2, 4, 6)

    return (
        batch_size,
        ncoil,
        klength,
        im_size_2d,
        im_size_3d,
        numpoints_2d,
        numpoints_3d,
    )


def create_input_plus_noise(shape, is_complex):
    x = np.arange(np.product(shape)).reshape(shape)
    x = torch.tensor(x, dtype=torch.get_default_dtype())

    if is_complex:
        x = x + torch.randn(size=x.shape) + 1j * torch.randn(size=x.shape)
    else:
        x = x + torch.randn(size=x.shape)

    return x


def create_ktraj(ndims, klength):
    ktraj = np.random.uniform(size=(ndims, klength)) * 2 * np.pi - np.pi

    return torch.tensor(ktraj, dtype=torch.get_default_dtype())
