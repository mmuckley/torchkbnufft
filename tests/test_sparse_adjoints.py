import sys

import numpy as np
import torch

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft)
from torchkbnufft.math import inner_product
from torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats

norm_tol = 1e-10


def test_interp_2d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    x = np.random.normal(size=(nslice, ncoil) + im_size) + \
        1j*np.random.normal(size=(nslice, ncoil) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2)).to(dtype)

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 2, klength)).to(dtype)

    kbinterp_ob = KbInterpForw(
        im_size=(20, 25), grid_size=im_size, numpoints=(4, 6))
    adjkbinterp_ob = KbInterpBack(
        im_size=(20, 25), grid_size=im_size, numpoints=(4, 6))

    real_mat, imag_mat = precomp_sparse_mats(ktraj, kbinterp_ob)
    interp_mats = {
        'real_interp_mats': real_mat,
        'imag_interp_mats': imag_mat
    }

    x_forw = kbinterp_ob(x, ktraj, interp_mats)
    y_back = adjkbinterp_ob(y, ktraj, interp_mats)

    inprod1 = inner_product(y, x_forw, dim=2)
    inprod2 = inner_product(y_back, x, dim=2)

    assert torch.norm(inprod1 - inprod2) < norm_tol


def test_nufft_2d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    x = np.random.normal(size=(nslice, ncoil) + im_size) + \
        1j*np.random.normal(size=(nslice, ncoil) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2)).to(dtype)

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 2, klength)).to(dtype)

    kbnufft_ob = KbNufft(im_size=im_size, numpoints=(4, 6))
    adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=(4, 6))

    real_mat, imag_mat = precomp_sparse_mats(ktraj, kbnufft_ob)
    interp_mats = {
        'real_interp_mats': real_mat,
        'imag_interp_mats': imag_mat
    }

    x_forw = kbnufft_ob(x, ktraj, interp_mats)
    y_back = adjkbnufft_ob(y, ktraj, interp_mats)

    inprod1 = inner_product(y, x_forw, dim=2)
    inprod2 = inner_product(y_back, x, dim=2)

    assert torch.norm(inprod1 - inprod2) < norm_tol


def test_mrisensenufft_2d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (33, 24)
    klength = 112

    x = np.random.normal(size=(nslice, 1) + im_size) + \
        1j*np.random.normal(size=(nslice, 1) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2)).to(dtype)

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 2, klength)).to(dtype)

    smap_sz = (nslice, ncoil, 2) + im_size
    smap = torch.randn(*smap_sz).to(dtype)

    sensenufft_ob = MriSenseNufft(smap=smap, im_size=im_size)
    adjsensenufft_ob = AdjMriSenseNufft(smap=smap, im_size=im_size)

    real_mat, imag_mat = precomp_sparse_mats(ktraj, sensenufft_ob)
    interp_mats = {
        'real_interp_mats': real_mat,
        'imag_interp_mats': imag_mat
    }

    x_forw = sensenufft_ob(x, ktraj, interp_mats)
    y_back = adjsensenufft_ob(y, ktraj, interp_mats)

    inprod1 = inner_product(y, x_forw, dim=2)
    inprod2 = inner_product(y_back, x, dim=2)

    assert torch.norm(inprod1 - inprod2) < norm_tol


def test_interp_3d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    x = np.random.normal(size=(nslice, ncoil) + im_size) + \
        1j*np.random.normal(size=(nslice, ncoil) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2)).to(dtype)

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 3, klength)).to(dtype)

    kbinterp_ob = KbInterpForw(
        im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6))
    adjkbinterp_ob = KbInterpBack(
        im_size=(5, 20, 25), grid_size=im_size, numpoints=(2, 4, 6))

    real_mat, imag_mat = precomp_sparse_mats(ktraj, kbinterp_ob)
    interp_mats = {
        'real_interp_mats': real_mat,
        'imag_interp_mats': imag_mat
    }

    x_forw = kbinterp_ob(x, ktraj, interp_mats)
    y_back = adjkbinterp_ob(y, ktraj, interp_mats)

    inprod1 = inner_product(y, x_forw, dim=2)
    inprod2 = inner_product(y_back, x, dim=2)

    assert torch.norm(inprod1 - inprod2) < norm_tol


def test_nufft_3d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    x = np.random.normal(size=(nslice, ncoil) + im_size) + \
        1j*np.random.normal(size=(nslice, ncoil) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2)).to(dtype)

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 3, klength)).to(dtype)

    kbnufft_ob = KbNufft(im_size=im_size, numpoints=(2, 4, 6))
    adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=(2, 4, 6))

    real_mat, imag_mat = precomp_sparse_mats(ktraj, kbnufft_ob)
    interp_mats = {
        'real_interp_mats': real_mat,
        'imag_interp_mats': imag_mat
    }

    x_forw = kbnufft_ob(x, ktraj, interp_mats)
    y_back = adjkbnufft_ob(y, ktraj, interp_mats)

    inprod1 = inner_product(y, x_forw, dim=2)
    inprod2 = inner_product(y_back, x, dim=2)

    assert torch.norm(inprod1 - inprod2) < norm_tol


def test_mrisensenufft_3d_adjoint():
    dtype = torch.double

    nslice = 2
    ncoil = 4
    im_size = (11, 33, 24)
    klength = 112

    x = np.random.normal(size=(nslice, 1) + im_size) + \
        1j*np.random.normal(size=(nslice, 1) + im_size)
    x = torch.tensor(np.stack((np.real(x), np.imag(x)), axis=2)).to(dtype)

    y = np.random.normal(size=(nslice, ncoil, klength)) + \
        1j*np.random.normal(size=(nslice, ncoil, klength))
    y = torch.tensor(np.stack((np.real(y), np.imag(y)), axis=2)).to(dtype)

    ktraj = torch.randn(*(nslice, 3, klength)).to(dtype)

    smap_sz = (nslice, ncoil, 2) + im_size
    smap = torch.randn(*smap_sz).to(dtype)

    sensenufft_ob = MriSenseNufft(smap=smap, im_size=im_size)
    adjsensenufft_ob = AdjMriSenseNufft(smap=smap, im_size=im_size)

    real_mat, imag_mat = precomp_sparse_mats(ktraj, sensenufft_ob)
    interp_mats = {
        'real_interp_mats': real_mat,
        'imag_interp_mats': imag_mat
    }

    x_forw = sensenufft_ob(x, ktraj, interp_mats)
    y_back = adjsensenufft_ob(y, ktraj, interp_mats)

    inprod1 = inner_product(y, x_forw, dim=2)
    inprod2 = inner_product(y_back, x, dim=2)

    assert torch.norm(inprod1 - inprod2) < norm_tol
