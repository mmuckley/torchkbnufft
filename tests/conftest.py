import pytest
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
        numpoints_3d
    )


@pytest.fixture
def device_list():
    return [torch.device('cpu'), torch.device('cuda')]


@pytest.fixture
def testing_dtype():
    return torch.double


@pytest.fixture
def testing_tol():
    return 1e-9


@pytest.fixture
def params_2d():
    batch_size, ncoil, klength, im_size, _, numpoints, _ = test_params()

    x = np.random.normal(size=(batch_size, 1) + im_size) + \
        1j*np.random.normal(size=(batch_size, 1) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2))

    y = np.random.normal(size=(batch_size, ncoil, klength)) + \
        1j*np.random.normal(size=(batch_size, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2))

    ktraj = (torch.rand(*(batch_size, 2, klength)) - 0.5) * 2*np.pi

    smap_sz = (batch_size, ncoil, 2) + im_size
    smap = torch.randn(*smap_sz)

    return {
        'batch_size': batch_size,
        'ncoil': ncoil,
        'klength': klength,
        'im_size': im_size,
        'grid_size': tuple(2 * np.array(im_size)),
        'numpoints': numpoints,
        'x': x,
        'y': y,
        'ktraj': ktraj,
        'smap': smap
    }


@pytest.fixture
def params_3d():
    batch_size, ncoil, klength, _, im_size, _, numpoints = test_params()

    x = np.random.normal(size=(batch_size, 1) + im_size) + \
        1j*np.random.normal(size=(batch_size, 1) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2))

    y = np.random.normal(size=(batch_size, ncoil, klength)) + \
        1j*np.random.normal(size=(batch_size, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2))

    ktraj = (torch.rand(*(batch_size, 3, klength)) - 0.5) * 2*np.pi
    coilpack_ktraj = (torch.rand(*(1, 2, klength)) - 0.5) * 2*np.pi

    smap_sz = (batch_size, ncoil, 2) + im_size
    smap = torch.randn(*smap_sz)

    return {
        'batch_size': batch_size,
        'ncoil': ncoil,
        'klength': klength,
        'im_size': im_size,
        'grid_size': tuple(2 * np.array(im_size)),
        'numpoints': numpoints,
        'x': x,
        'y': y,
        'ktraj': ktraj,
        'coilpack_ktraj': coilpack_ktraj,
        'smap': smap
    }
