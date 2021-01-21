import numpy as np
import torch
import torchkbnufft as tkbn


def test_complex_mult():
    x = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    y = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    x_torch = torch.stack((torch.tensor(np.real(x)), torch.tensor(np.imag(x))), -1)
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))), -1)

    z = x * y

    z_torch = tkbn.complex_mult(x_torch, y_torch).numpy()
    z_torch = z_torch[..., 0] + 1j * z_torch[..., 1]

    assert np.allclose(z, z_torch)

    x_torch = x_torch = torch.stack(
        (torch.tensor(np.real(x)), torch.tensor(np.imag(x))), dim=2
    )
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))), dim=2)
    z_torch = tkbn.complex_mult(x_torch, y_torch, dim=2).numpy()
    z_torch = z_torch[:, :, 0, ...] + 1j * z_torch[:, :, 1, ...]

    assert np.allclose(z, z_torch)


def test_conj_complex_mult():
    x = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    y = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    x_torch = torch.stack((torch.tensor(np.real(x)), torch.tensor(np.imag(x))), dim=-1)
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))), dim=-1)

    z = x * np.conj(y)

    z_torch = tkbn.conj_complex_mult(x_torch, y_torch).numpy()
    z_torch = z_torch[..., 0] + 1j * z_torch[..., 1]

    assert np.allclose(z, z_torch)

    x_torch = x_torch = torch.stack(
        (torch.tensor(np.real(x)), torch.tensor(np.imag(x))), dim=2
    )
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))), dim=2)
    z_torch = tkbn.conj_complex_mult(x_torch, y_torch, dim=2).numpy()
    z_torch = z_torch[:, :, 0, ...] + 1j * z_torch[:, :, 1, ...]

    assert np.allclose(z, z_torch)
