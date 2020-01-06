import sys

import numpy as np
import torch

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft, ToepNufft,
                          ToepSenseNufft)
from torchkbnufft.math import inner_product
from torchkbnufft.nufft.toep_functions import calc_toep_kernel


def test_interp_2d_adjoint(params_2d, testing_tol, testing_dtype, device_list):
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

        x_forw = kbinterp_ob(x, ktraj)
        y_back = adjkbinterp_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_nufft_2d_adjoint(params_2d, testing_tol, testing_dtype, device_list):
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

        x_forw = kbnufft_ob(x, ktraj)
        y_back = adjkbnufft_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_toepnufft_2d_adjoint(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = torch.randn(x.shape)
    ktraj = params_2d['ktraj']
    smap = params_2d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        adjkbnufft_ob = AdjKbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        toep_ob = ToepNufft().to(dtype=dtype, device=device)

        kern = calc_toep_kernel(adjkbnufft_ob, ktraj)

        x_forw = toep_ob(x, kern)
        y_back = toep_ob(y, kern)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_mrisensenufft_2d_adjoint(params_2d, testing_tol, testing_dtype, device_list):
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

        x_forw = sensenufft_ob(x, ktraj)
        y_back = adjsensenufft_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_toepsensenufft_2d_adjoint(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_2d['im_size']
    numpoints = params_2d['numpoints']

    x = params_2d['x']
    y = torch.randn(x.shape)
    ktraj = params_2d['ktraj']
    smap = params_2d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        toep_ob = ToepSenseNufft(smap=smap).to(dtype=dtype, device=device)

        kern = calc_toep_kernel(adjsensenufft_ob, ktraj)

        x_forw = toep_ob(x, kern)
        y_back = toep_ob(y, kern)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_interp_3d_adjoint(params_3d, testing_tol, testing_dtype, device_list):
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

        x_forw = kbinterp_ob(x, ktraj)
        y_back = adjkbinterp_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_nufft_3d_adjoint(params_3d, testing_tol, testing_dtype, device_list):
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

        x_forw = kbnufft_ob(x, ktraj)
        y_back = adjkbnufft_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_toepnufft_3d_adjoint(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_3d['im_size']
    numpoints = params_3d['numpoints']

    x = params_3d['x']
    y = torch.randn(x.shape)
    ktraj = params_3d['ktraj']
    smap = params_3d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        adjkbnufft_ob = AdjKbNufft(
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        toep_ob = ToepNufft().to(dtype=dtype, device=device)

        kern = calc_toep_kernel(adjkbnufft_ob, ktraj)

        x_forw = toep_ob(x, kern)
        y_back = toep_ob(y, kern)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_mrisensenufft_3d_adjoint(params_3d, testing_tol, testing_dtype, device_list):
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

        x_forw = sensenufft_ob(x, ktraj)
        y_back = adjsensenufft_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_toepsensenufft_3d_adjoint(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    norm_tol = testing_tol

    im_size = params_3d['im_size']
    numpoints = params_3d['numpoints']

    x = params_3d['x']
    y = torch.randn(x.shape)
    ktraj = params_3d['ktraj']
    smap = params_3d['smap']

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap,
            im_size=im_size,
            numpoints=numpoints
        ).to(dtype=dtype, device=device)
        toep_ob = ToepSenseNufft(smap=smap).to(dtype=dtype, device=device)

        kern = calc_toep_kernel(adjsensenufft_ob, ktraj)

        x_forw = toep_ob(x, kern)
        y_back = toep_ob(y, kern)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol


def test_mrisensenufft_3d_coilpack_adjoint(params_2d, testing_tol, testing_dtype, device_list):
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

        x_forw = sensenufft_ob(x, ktraj)
        y_back = adjsensenufft_ob(y, ktraj)

        inprod1 = inner_product(y, x_forw, dim=2)
        inprod2 = inner_product(y_back, x, dim=2)

        assert torch.norm(inprod1 - inprod2) < norm_tol
