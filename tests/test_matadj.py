import sys

import numpy as np
import torch

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft)
from torchkbnufft.math import inner_product

norm_tol = 1e-10


def test_interp_2d_matadj():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 2, klength)).to(dtype)

    adjkbinterp_ob = KbInterpBack(
        im_size=(20, 25), grid_size=im_size, numpoints=(4, 6))
    adjkbinterp_matadj_ob = KbInterpBack(
        im_size=(20, 25), grid_size=im_size, numpoints=(4, 6), matadj=True)

    x_normal = adjkbinterp_ob(y, ktraj)
    x_matadj = adjkbinterp_matadj_ob(y, ktraj)

    assert torch.norm(x_normal - x_matadj) / torch.norm(x_normal) < norm_tol


def test_nufft_2d_matadj():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 2, klength)).to(dtype)

    adjkbnufft_ob = AdjKbNufft(
        im_size=im_size, numpoints=(4, 6))
    adjkbnufft_matadj_ob = AdjKbNufft(
        im_size=im_size, numpoints=(4, 6), matadj=True)

    x_normal = adjkbnufft_ob(y, ktraj)
    x_matadj = adjkbnufft_matadj_ob(y, ktraj)

    assert torch.norm(x_normal - x_matadj) / torch.norm(x_normal) < norm_tol


def test_mrisensenufft_2d_matadj():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 2, klength)).to(dtype)

    smap_sz = (nslice, ncoil, 2) + im_size
    smap = torch.randn(*smap_sz).to(dtype)

    adjsensenufft_ob = AdjMriSenseNufft(smap=smap, im_size=im_size)
    adjsensenufft_matadj_ob = AdjMriSenseNufft(
        smap=smap, im_size=im_size, matadj=True)

    x_normal = adjsensenufft_ob(y, ktraj)
    x_matadj = adjsensenufft_matadj_ob(y, ktraj)

    assert torch.norm(x_normal - x_matadj) / torch.norm(x_normal) < norm_tol


def test_interp_3d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 3, klength)).to(dtype)

    adjkbinterp_ob = KbInterpBack(
        im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6))
    adjkbinterp_matadj_ob = KbInterpBack(
        im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6), matadj=True)

    x_normal = adjkbinterp_ob(y, ktraj)
    x_matadj = adjkbinterp_matadj_ob(y, ktraj)

    assert torch.norm(x_normal - x_matadj) < norm_tol


def test_nufft_3d_matadj():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 3, klength)).to(dtype)

    adjkbnufft_ob = AdjKbNufft(
        im_size=im_size, numpoints=(2, 4, 6))
    adjkbnufft_matadj_ob = AdjKbNufft(
        im_size=im_size, numpoints=(2, 4, 6), matadj=True)

    x_normal = adjkbnufft_ob(y, ktraj)
    x_matadj = adjkbnufft_matadj_ob(y, ktraj)

    assert torch.norm(x_normal - x_matadj) / torch.norm(x_normal) < norm_tol


def test_mrisensenufft_3d_matadj():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 3, klength)).to(dtype)

    smap_sz = (nslice, ncoil, 2) + im_size
    smap = torch.randn(*smap_sz).to(dtype)

    adjsensenufft_ob = AdjMriSenseNufft(smap=smap, im_size=im_size)
    adjsensenufft_matadj_ob = AdjMriSenseNufft(
        smap=smap, im_size=im_size, matadj=True)

    x_normal = adjsensenufft_ob(y, ktraj)
    x_matadj = adjsensenufft_matadj_ob(y, ktraj)

    assert torch.norm(x_normal - x_matadj) / torch.norm(x_normal) < norm_tol
