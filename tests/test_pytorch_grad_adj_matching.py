import sys

import numpy as np
import torch

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft)


def test_2d_interp_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    batch_size = params_2d['batch_size']
    im_size = params_2d['im_size']
    grid_size = params_2d['grid_size']
    numpoints = params_2d['numpoints']

    x = np.random.normal(size=(batch_size, 1) + grid_size) + \
        1j*np.random.normal(size=(batch_size, 1) + grid_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2))
    y = params_2d['y']
    ktraj = params_2d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = kbinterp_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbinterp_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_interp_adjoint_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    batch_size = params_2d['batch_size']
    im_size = params_2d['im_size']
    grid_size = params_2d['grid_size']
    numpoints = params_2d['numpoints']

    x = np.random.normal(size=(batch_size, 1) + grid_size) + \
        1j*np.random.normal(size=(batch_size, 1) + grid_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2))
    y = params_2d['y']
    ktraj = params_2d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbinterp_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbinterp_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_2d_kbnufft_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = params_2d['y']
    ktraj = params_2d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbnufft_ob = KbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = kbnufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbnufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_kbnufft_adjoint_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = params_2d['y']
    ktraj = params_2d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbnufft_ob = KbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbnufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbnufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_2d_mrisensenufft_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = params_2d['y']
    ktraj = params_2d['ktraj']
    smap = params_2d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = sensenufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_mrisensenufft_adjoint_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = params_2d['y']
    ktraj = params_2d['ktraj']
    smap = params_2d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjsensenufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_interp_backward(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    batch_size = params_3d['batch_size']
    im_size = params_3d['im_size']
    grid_size = params_3d['grid_size']
    numpoints = params_3d['numpoints']

    x = np.random.normal(size=(batch_size, 1) + grid_size) + \
        1j*np.random.normal(size=(batch_size, 1) + grid_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2))
    y = params_3d['y']
    ktraj = params_3d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = kbinterp_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbinterp_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_interp_adjoint_backward(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    batch_size = params_3d['batch_size']
    im_size = params_3d['im_size']
    grid_size = params_3d['grid_size']
    numpoints = params_3d['numpoints']

    x = np.random.normal(size=(batch_size, 1) + grid_size) + \
        1j*np.random.normal(size=(batch_size, 1) + grid_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2))
    y = params_3d['y']
    ktraj = params_3d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbinterp_ob = KbInterpForw(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbinterp_ob = KbInterpBack(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbinterp_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbinterp_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_kbnufft_backward(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_3d['im_size']
    numpoints = params_3d['numpoints']

    x = params_3d['x']
    y = params_3d['y']
    ktraj = params_3d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbnufft_ob = KbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = kbnufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjkbnufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_kbnufft_adjoint_backward(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_3d['im_size']
    numpoints = params_3d['numpoints']

    x = params_3d['x']
    y = params_3d['y']
    ktraj = params_3d['ktraj']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbnufft_ob = KbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjkbnufft_ob = AdjKbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjkbnufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = kbnufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_mrisensenufft_backward(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_3d['im_size']
    numpoints = params_3d['numpoints']

    x = params_3d['x']
    y = params_3d['y']
    ktraj = params_3d['ktraj']
    smap = params_3d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = sensenufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_mrisensenufft_adjoint_backward(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_3d['im_size']
    numpoints = params_3d['numpoints']

    x = params_3d['x']
    y = params_3d['y']
    ktraj = params_3d['ktraj']
    smap = params_3d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjsensenufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_mrisensenufft_coilpack_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = params_2d['y']
    ktraj = params_2d['ktraj']
    smap = params_2d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints,
            coilpack=True
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints,
            coilpack=True
        ).to(dtype=dtype, device=device)

        x.requires_grad = True
        y = sensenufft_ob.forward(x, ktraj)

        ((y ** 2) / 2).sum().backward()
        x_grad = x.grad.clone().detach()

        x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj)

        assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_mrisensenufft_coilpack_adjoint_backward(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = params_2d['y']
    ktraj = params_2d['ktraj']
    smap = params_2d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints,
            coilpack=True
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints,
            coilpack=True
        ).to(dtype=dtype, device=device)

        y.requires_grad = True
        x = adjsensenufft_ob.forward(y, ktraj)

        ((x ** 2) / 2).sum().backward()
        y_grad = y.grad.clone().detach()

        y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj)

        assert torch.norm(y_grad-y_hat) < norm_tol
