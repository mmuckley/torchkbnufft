import sys

import numpy as np
import torch

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft)
from torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats

norm_tol = 1e-9


def test_2d_interp_backward():
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

    x.requires_grad = True
    y = kbinterp_ob.forward(x, ktraj, interp_mats)

    ((y ** 2) / 2).sum().backward()
    x_grad = x.grad.clone().detach()

    x_hat = adjkbinterp_ob.forward(y.clone().detach(), ktraj, interp_mats)

    assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_interp_adjoint_backward():
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

    y.requires_grad = True
    x = adjkbinterp_ob.forward(y, ktraj, interp_mats)

    ((x ** 2) / 2).sum().backward()
    y_grad = y.grad.clone().detach()

    y_hat = kbinterp_ob.forward(x.clone().detach(), ktraj, interp_mats)

    assert torch.norm(y_grad-y_hat) < norm_tol


def test_2d_kbnufft_backward():
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

    x.requires_grad = True
    y = kbnufft_ob.forward(x, ktraj, interp_mats)

    ((y ** 2) / 2).sum().backward()
    x_grad = x.grad.clone().detach()

    x_hat = adjkbnufft_ob.forward(y.clone().detach(), ktraj, interp_mats)

    assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_kbnufft_adjoint_backward():
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

    y.requires_grad = True
    x = adjkbnufft_ob.forward(y, ktraj, interp_mats)

    ((x ** 2) / 2).sum().backward()
    y_grad = y.grad.clone().detach()

    y_hat = kbnufft_ob.forward(x.clone().detach(), ktraj, interp_mats)

    assert torch.norm(y_grad-y_hat) < norm_tol


def test_2d_mrisensenufft_backward():
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

    x.requires_grad = True
    y = sensenufft_ob.forward(x, ktraj, interp_mats)

    ((y ** 2) / 2).sum().backward()
    x_grad = x.grad.clone().detach()

    x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj, interp_mats)

    assert torch.norm(x_grad-x_hat) < norm_tol


def test_2d_mrisensenufft_adjoint_backward():
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

    y.requires_grad = True
    x = adjsensenufft_ob.forward(y, ktraj, interp_mats)

    ((x ** 2) / 2).sum().backward()
    y_grad = y.grad.clone().detach()

    y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj, interp_mats)

    assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_interp_backward():
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

    x.requires_grad = True
    y = kbinterp_ob.forward(x, ktraj, interp_mats)

    ((y ** 2) / 2).sum().backward()
    x_grad = x.grad.clone().detach()

    x_hat = adjkbinterp_ob.forward(y.clone().detach(), ktraj, interp_mats)

    assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_interp_adjoint_backward():
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

    y.requires_grad = True
    x = adjkbinterp_ob.forward(y, ktraj, interp_mats)

    ((x ** 2) / 2).sum().backward()
    y_grad = y.grad.clone().detach()

    y_hat = kbinterp_ob.forward(x.clone().detach(), ktraj, interp_mats)

    assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_kbnufft_backward():
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

    x.requires_grad = True
    y = kbnufft_ob.forward(x, ktraj, interp_mats)

    ((y ** 2) / 2).sum().backward()
    x_grad = x.grad.clone().detach()

    x_hat = adjkbnufft_ob.forward(y.clone().detach(), ktraj, interp_mats)

    assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_kbnufft_adjoint_backward():
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

    y.requires_grad = True
    x = adjkbnufft_ob.forward(y, ktraj, interp_mats)

    ((x ** 2) / 2).sum().backward()
    y_grad = y.grad.clone().detach()

    y_hat = kbnufft_ob.forward(x.clone().detach(), ktraj, interp_mats)

    assert torch.norm(y_grad-y_hat) < norm_tol


def test_3d_mrisensenufft_backward():
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

    x.requires_grad = True
    y = sensenufft_ob.forward(x, ktraj, interp_mats)

    ((y ** 2) / 2).sum().backward()
    x_grad = x.grad.clone().detach()

    x_hat = adjsensenufft_ob.forward(y.clone().detach(), ktraj, interp_mats)

    assert torch.norm(x_grad-x_hat) < norm_tol


def test_3d_mrisensenufft_adjoint_backward():
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

    y.requires_grad = True
    x = adjsensenufft_ob.forward(y, ktraj, interp_mats)

    ((x ** 2) / 2).sum().backward()
    y_grad = y.grad.clone().detach()

    y_hat = sensenufft_ob.forward(x.clone().detach(), ktraj, interp_mats)

    assert torch.norm(y_grad-y_hat) < norm_tol


if __name__ == '__main__':
    test_3d_mrisensenufft_adjoint_backward()
