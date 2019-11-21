import sys

import numpy as np
import torch

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft)

norm_tol = 1e-10
devices = [torch.device('cuda'), torch.device('cpu')]


def test_2d_interp_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 2, klength),
                            dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=(20, 25), grid_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=(20, 25), grid_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)

        x.requires_grad = True
        y = kbinterp_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbinterp_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_interp_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 2, klength),
                            dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=(20, 25), grid_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=(20, 25), grid_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbinterp_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbinterp_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_2d_kbnufft_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 2, klength),
                            dtype=dtype, device=device)

        kbnufft_ob = KbNufft(im_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)

        x.requires_grad = True
        y = kbnufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbnufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_kbnufft_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 2, klength),
                            dtype=dtype, device=device)

        kbnufft_ob = KbNufft(im_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=(4, 6)).to(
            dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbnufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbnufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_2d_mrisensenufft_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 2, klength),
                            dtype=dtype, device=device)

        smap_sz = (nslice, ncoil, 2) + im_size
        smap = torch.randn(*smap_sz, dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = sensenufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_mrisensenufft_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 2, klength),
                            dtype=dtype, device=device)

        smap_sz = (nslice, ncoil, 2) + im_size
        smap = torch.randn(*smap_sz, dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjsensenufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_interp_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 3, klength),
                            dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)

        x.requires_grad = True
        y = kbinterp_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbinterp_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_interp_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 3, klength),
                            dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbinterp_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbinterp_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_kbnufft_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 3, klength),
                            dtype=dtype, device=device)

        kbnufft_ob = KbNufft(im_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)

        x.requires_grad = True
        y = kbnufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbnufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_kbnufft_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 3, klength),
                            dtype=dtype, device=device)

        kbnufft_ob = KbNufft(im_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=(2, 4, 6)).to(
            dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbnufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbnufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_mrisensenufft_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 3, klength),
                            dtype=dtype, device=device)

        smap_sz = (nslice, ncoil, 2) + im_size
        smap = torch.randn(*smap_sz, dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = sensenufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_mrisensenufft_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(nslice, 3, klength),
                            dtype=dtype, device=device)

        smap_sz = (nslice, ncoil, 2) + im_size
        smap = torch.randn(*smap_sz, dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjsensenufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_mrisensenufft_coilpack_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(1, 3, klength),
                            dtype=dtype, device=device)

        smap_sz = (nslice, ncoil, 2) + im_size
        smap = torch.randn(*smap_sz, dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size, coilpack=True).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size, coilpack=True).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = sensenufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_mrisensenufft_coilpack_adjoint_backward():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    for device in devices:
        x = np.random.normal(size=(nslice, 1) + im_size) + \
            1j*np.random.normal(size=(nslice, 1) + im_size)
        x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2),
                         dtype=dtype, device=device)

        y = np.random.normal(size=(nslice, ncoil, klength)) + \
            1j*np.random.normal(size=(nslice, ncoil, klength))
        y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2),
                         dtype=dtype, device=device)

        ktraj = torch.randn(*(1, 3, klength),
                            dtype=dtype, device=device)

        smap_sz = (nslice, ncoil, 2) + im_size
        smap = torch.randn(*smap_sz, dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size, coilpack=True).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size, coilpack=True).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjsensenufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol
