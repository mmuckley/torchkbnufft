import sys

import numpy as np
import torch
from torchkbnufft import (
    AdjKbNufft,
    AdjMriSenseNufft,
    KbNufft,
    MriSenseNufft,
    ToepNufft,
    ToepSenseNufft,
)
from torchkbnufft.math import inner_product
from torchkbnufft.nufft.toep_functions import calc_toep_kernel


def test_toeplitz_nufft_2d(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    # this tolerance looks really bad, but toep struggles with random traj
    # for radial it's more like 1e-06
    norm_tol = 1e-3

    im_size = params_2d["im_size"]
    numpoints = params_2d["numpoints"]

    x = params_2d["x"]
    y = params_2d["y"]
    ktraj = params_2d["ktraj"]

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbnufft_ob = KbNufft(im_size=im_size, numpoints=numpoints).to(
            dtype=dtype, device=device
        )
        adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=numpoints).to(
            dtype=dtype, device=device
        )
        toep_ob = ToepNufft()

        kern = calc_toep_kernel(adjkbnufft_ob, ktraj)

        normal_forw = adjkbnufft_ob(kbnufft_ob(x, ktraj), ktraj)

        toep_forw = toep_ob(x, kern)

        diff = torch.norm(normal_forw - toep_forw) / torch.norm(normal_forw)

        assert diff < norm_tol


def test_toeplitz_mrisensenufft_2d(params_2d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    # this tolerance looks really bad, but toep struggles with random traj
    # for radial it's more like 1e-06
    norm_tol = 1e-3

    im_size = params_2d["im_size"]
    numpoints = params_2d["numpoints"]

    x = params_2d["x"]
    y = params_2d["y"]
    ktraj = params_2d["ktraj"]
    smap = params_2d["smap"]

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size, numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size, numpoints=numpoints
        ).to(dtype=dtype, device=device)
        toep_ob = ToepSenseNufft(smap=smap).to(dtype=dtype, device=device)

        kern = calc_toep_kernel(adjsensenufft_ob, ktraj)

        normal_forw = adjsensenufft_ob(sensenufft_ob(x, ktraj), ktraj)

        toep_forw = toep_ob(x, kern)

        diff = torch.norm(normal_forw - toep_forw) / torch.norm(normal_forw)

        assert diff < norm_tol


def test_toeplitz_nufft_3d(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    # this tolerance looks really bad, but toep struggles with random traj
    # for radial it's more like 1e-06
    norm_tol = 1e-1

    im_size = params_3d["im_size"]
    numpoints = params_3d["numpoints"]

    x = params_3d["x"]
    y = params_3d["y"]
    ktraj = params_3d["ktraj"]

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        kbnufft_ob = KbNufft(im_size=im_size, numpoints=numpoints).to(
            dtype=dtype, device=device
        )
        adjkbnufft_ob = AdjKbNufft(im_size=im_size, numpoints=numpoints).to(
            dtype=dtype, device=device
        )
        toep_ob = ToepNufft()

        kern = calc_toep_kernel(adjkbnufft_ob, ktraj)

        normal_forw = adjkbnufft_ob(kbnufft_ob(x, ktraj), ktraj)

        toep_forw = toep_ob(x, kern)

        diff = torch.norm(normal_forw - toep_forw) / torch.norm(normal_forw)

        assert diff < norm_tol


def test_toeplitz_mrisensenufft_3d(params_3d, testing_tol, testing_dtype, device_list):
    dtype = testing_dtype
    # this tolerance looks really bad, but toep struggles with random traj
    # for radial it's more like 1e-06
    norm_tol = 1e-1

    im_size = params_3d["im_size"]
    numpoints = params_3d["numpoints"]

    x = params_3d["x"]
    y = params_3d["y"]
    ktraj = params_3d["ktraj"]
    smap = params_3d["smap"]

    for device in device_list:
        x = x.detach().to(dtype=dtype, device=device)
        y = y.detach().to(dtype=dtype, device=device)
        ktraj = ktraj.detach().to(dtype=dtype, device=device)

        sensenufft_ob = MriSenseNufft(
            smap=smap, im_size=im_size, numpoints=numpoints
        ).to(dtype=dtype, device=device)
        adjsensenufft_ob = AdjMriSenseNufft(
            smap=smap, im_size=im_size, numpoints=numpoints
        ).to(dtype=dtype, device=device)
        toep_ob = ToepSenseNufft(smap=smap).to(dtype=dtype, device=device)

        kern = calc_toep_kernel(adjsensenufft_ob, ktraj)

        normal_forw = adjsensenufft_ob(sensenufft_ob(x, ktraj), ktraj)

        toep_forw = toep_ob(x, kern)

        diff = torch.norm(normal_forw - toep_forw) / torch.norm(normal_forw)

        assert diff < norm_tol
