import copy

import numpy as np
import torch

from ..nufft.interp_functions import kbinterp, adjkbinterp


def calculate_radial_dcomp_pytorch(nufftob_forw, nufftob_back, ktraj, stacks=True):
    """Numerical density compensation estimation for a radial trajectory.

    Estimates the density compensation function numerically using a NUFFT
    operator (nufftob_forw and nufftob_back) and a k-space trajectory (ktraj).
    The function applies A'A1 (where A is the nufftob and 1 is a ones vector)
    and estimates the signal accumulation at the origin of k-space. It then
    returns a vector of density compensation values that are computed based on
    the distance from the k-space center and thresholded above the center
    density estimate. Then, a density-compensated image can be calculated by
    applying A'Wy, where W is a diagonal matrix with the density compensation
    values.

    This function uses a PyTorch NUFFT object and k-space trajectory.

    Args:
        nufftob (object): A PyTorch NUFFT object.
        traj (tensor or list of tensors): The k-space trajectory in
            radians/voxel of length b, where each of the trajectories in the
            list has dimension (d, m). In this notation, b is a batch
            dimension, d is the number of spatial dimensions, and m is the
            length of the trajectory. Trajectories of different sizes can be
            passed in using a list.
        stacks (bool): whether the trajectory is actually a stacks of radial
            for 3D imaging rather than a pure radial trajectory. The stacks
            dimension must be the first dimension of the trajectory.
            Defaults to True.

    Returns:
        tensor or list of tensors: The density compensation coefficients for
            ktraj of size (m). If b == 1, then dcomps is a tensor with a batch
            size of 1. If b > 1, then dcomps is a list. If all input
            trajectories are of the same size, then torch.stack(dcomps) can be
            used to get an array of size (b, m).

    """
    dtype = nufftob_forw.scaling_coef_tensor.dtype
    device = nufftob_forw.scaling_coef_tensor.device

    nufftob_forw = copy.deepcopy(nufftob_forw).to(dtype=dtype, device=device)
    nufftob_back = copy.deepcopy(nufftob_back).to(dtype=dtype, device=device)

    # remove sensitivities if dealing with MriSenseNufft
    if "Sense" in nufftob_forw.__class__.__name__:
        nufftob_forw = copy.deepcopy(nufftob_forw)
        nufftob_back = copy.deepcopy(nufftob_back)

        nufftob_forw.smap_tensor = torch.ones(
            nufftob_forw.smap_tensor.shape, dtype=dtype, device=device
        )
        nufftob_forw.smap_tensor = nufftob_forw.smap_tensor[:, 0:1]
        nufftob_forw.smap_tensor[:, :, 1] = 0

        nufftob_back.smap_tensor = nufftob_forw.smap_tensor.clone()

    if not nufftob_forw.norm == "ortho":
        if not nufftob_back.norm == "ortho":
            norm_factor = torch.prod(torch.tensor(nufftob_back.grid_size)).to(
                dtype=dtype, device=device
            )
        else:
            print("warning: forward/backward operators mismatched norm setting")
            norm_factor = 1
    elif not nufftob_back.norm == "ortho":
        print("warning: forward/backward operators mismatched norm setting")
        norm_factor = 1
    else:
        norm_factor = 1

    # append 0s for batch, first coil, real part
    image_loc = (
        0,
        0,
        0,
    ) + tuple((np.array(nufftob_forw.im_size) // 2).astype(np.int))

    # get the size of the test signal (add batch, coil, real/imag dim)
    test_size = (1, 1, 2) + nufftob_forw.im_size

    test_sig = torch.ones(test_size, dtype=dtype, device=device)
    dcomps = []

    # get one dcomp for each batch
    threshold_levels = torch.zeros(len(ktraj), dtype=dtype, device=device)
    for batch_ind, batch_traj in enumerate(ktraj):
        # extract the signal amplitude increase from center of image
        query_point = (
            nufftob_back(
                nufftob_forw(test_sig, om=batch_traj.unsqueeze(0)),
                om=batch_traj.unsqueeze(0),
            )[image_loc]
            / norm_factor
        )

        # use query point to get ramp intercept
        threshold_levels[batch_ind] = 1 / query_point

        # compute the new dcomp for the batch in batch_ind
        ndims = len(nufftob_forw.im_size)
        if stacks:
            batch_traj_thresh = batch_traj[-2:]
        else:
            batch_traj_thresh = batch_traj[-ndims:]
        dcomps.append(
            torch.max(
                torch.sqrt(torch.sum(batch_traj_thresh ** 2, dim=0)) * 1 / np.pi,
                threshold_levels[batch_ind],
            )
        )

    if isinstance(ktraj, torch.Tensor):
        dcomps = torch.stack(dcomps)

    return dcomps


def calculate_density_compensator(interpob, ktraj, num_iterations=10):
    """Numerical density compensation estimation for a any trajectory.

    Estimates the density compensation function numerically using a NUFFT
    interpolator operator and a k-space trajectory (ktraj).
    This function implements Pipe et al

    This function uses a nufft hyper parameter dictionary, the associated nufft
    operators and k-space trajectory.

    Args:
        interpob (dict): the output of `KbNufftModule._extract_nufft_interpob`
            containing all the hyper-parameters for the nufft computation.
        ktraj (tensor): The k-space trajectory in radians/voxel dimension (d, m).
            d is the number of spatial dimensions, and m is the length of the
            trajectory.
        num_iterations (int): default 10
            number of iterations

    Returns:
        tensor: The density compensation coefficients for ktraj of size (m).
    """
    test_sig = torch.ones([1, 1, ktraj.shape[-1]])
    test_sig = torch.stack([test_sig, torch.zeros_like(test_sig)], 2)
    for i in range(num_iterations):
        new_sig = kbinterp(
            adjkbinterp(test_sig, ktraj, interpob),
            ktraj,
            interpob
        )
        # Basically we are doing abs here, do we have utils for this?
        norm_new_sig = torch.norm(new_sig, dim=2)
        test_sig = test_sig / norm_new_sig
    return test_sig[0][0][0].unsqueeze(0)