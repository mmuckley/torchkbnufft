import numpy as np
import torch
from torchkbnufft.math import complex_mult, conj_complex_mult


def test_complex_mult():
    x = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    y = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    x_torch = torch.stack((torch.tensor(np.real(x)), torch.tensor(np.imag(x))))
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))))

    z = x * y

    z_torch = complex_mult(x_torch, y_torch).numpy()
    z_torch = z_torch[0, ...] + 1j * z_torch[1, ...]

    assert np.linalg.norm(z - z_torch) / np.linalg.norm(z) < 1e-12

    x_torch = x_torch = torch.stack(
        (torch.tensor(np.real(x)), torch.tensor(np.imag(x))), dim=2
    )
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))), dim=2)
    z_torch = complex_mult(x_torch, y_torch, dim=2).numpy()
    z_torch = z_torch[:, :, 0, ...] + 1j * z_torch[:, :, 1, ...]

    assert np.linalg.norm(z - z_torch) / np.linalg.norm(z) < 1e-12


def test_conj_complex_mult():
    x = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    y = np.random.normal(size=(8, 16, 10)) + 1j * np.random.normal(size=(8, 16, 10))
    x_torch = torch.stack((torch.tensor(np.real(x)), torch.tensor(np.imag(x))))
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))))

    z = x * np.conj(y)

    z_torch = conj_complex_mult(x_torch, y_torch).numpy()
    z_torch = z_torch[0, ...] + 1j * z_torch[1, ...]

    assert np.linalg.norm(z - z_torch) / np.linalg.norm(z) < 1e-12

    x_torch = x_torch = torch.stack(
        (torch.tensor(np.real(x)), torch.tensor(np.imag(x))), dim=2
    )
    y_torch = torch.stack((torch.tensor(np.real(y)), torch.tensor(np.imag(y))), dim=2)
    z_torch = conj_complex_mult(x_torch, y_torch, dim=2).numpy()
    z_torch = z_torch[:, :, 0, ...] + 1j * z_torch[:, :, 1, ...]

    assert np.linalg.norm(z - z_torch) / np.linalg.norm(z) < 1e-12
