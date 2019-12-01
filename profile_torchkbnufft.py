import time

import numpy as np
import torch
from PIL import Image
from skimage.data import camera

from torchkbnufft import AdjKbNufft, AdjMriSenseNufft, MriSenseNufft
from torchkbnufft.mri.mrisensesim import mrisensesim
from torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats


def profile_torchkbnufft(image, ktraj, smap, im_size, device, sparse_mats_flag=False):
    # run double precision for CPU, float for GPU
    # these seem to be present in reference implementations
    if device == torch.device('cpu'):
        dtype = torch.double
        num_nuffts = 5
    else:
        dtype = torch.float
        num_nuffts = 20
    cpudevice = torch.device('cpu')

    image = image.to(dtype=dtype)
    ktraj = ktraj.to(dtype=dtype)
    smap = smap.to(dtype=dtype)

    kbsense_ob = MriSenseNufft(smap=smap, im_size=im_size).to(
        dtype=dtype, device=device)
    adjkbsense_ob = AdjMriSenseNufft(
        smap=smap, im_size=im_size).to(dtype=dtype, device=device)

    adjkbnufft_ob = AdjKbNufft(im_size=im_size).to(dtype=dtype, device=device)

    # precompute the sparse interpolation matrices
    if sparse_mats_flag:
        print('using sparse interpolation matrices')
        real_mat, imag_mat = precomp_sparse_mats(ktraj, adjkbnufft_ob)
        interp_mats = {
            'real_interp_mats': real_mat,
            'imag_interp_mats': imag_mat
        }
    else:
        print('not using sparse interpolation matrices')
        interp_mats = None

    # warm-up computation
    for _ in range(num_nuffts):
        y = kbsense_ob(image.to(device=device), ktraj.to(
            device=device), interp_mats).to(cpudevice)

    # run the forward speed tests
    if device == torch.device('cuda'):
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_nuffts):
        y = kbsense_ob(image.to(device=device), ktraj.to(
            device=device), interp_mats)
    if device == torch.device('cuda'):
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()
        print('GPU forward max memory: {} GB'.format(max_mem/1e9))
    end_time = time.perf_counter()
    avg_time = (end_time-start_time) / num_nuffts
    print('forward average time: {}'.format(avg_time))

    # warm-up computation
    for _ in range(num_nuffts):
        x = adjkbsense_ob(y.to(device), ktraj.to(
            device), interp_mats)

    # run the adjoint speed tests
    if device == torch.device('cuda'):
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(num_nuffts):
        x = adjkbsense_ob(y.to(device), ktraj.to(
            device), interp_mats)
    if device == torch.device('cuda'):
        torch.cuda.synchronize()
        max_mem = torch.cuda.max_memory_allocated()
        print('GPU adjoint max memory: {} GB'.format(max_mem/1e9))
    end_time = time.perf_counter()
    avg_time = (end_time-start_time) / num_nuffts
    print('backward average time: {}'.format(avg_time))


def run_all_profiles():
    print('running profiler...')
    spokelength = 512
    nspokes = 405
    ncoil = 15

    print('problem size (radial trajectory, 2-factor oversampling):')
    print('number of coils: {}'.format(ncoil))
    print('number of spokes: {}'.format(nspokes))
    print('spokelength: {}'.format(spokelength))

    # create an example to run on
    image = np.array(Image.fromarray(camera()).resize((256, 256)))
    image = image.astype(np.complex)
    im_size = image.shape

    image = np.stack((np.real(image), np.imag(image)))
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)

    # create k-space trajectory
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros(shape=(spokelength, nspokes))
    ky = np.zeros(shape=(spokelength, nspokes))
    ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
        ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

    ky = np.transpose(ky)
    kx = np.transpose(kx)

    ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)

    ktraj = torch.tensor(ktraj).unsqueeze(0)

    smap_sz = (1, ncoil, 2) + im_size
    smap = torch.ones(*smap_sz)

    profile_torchkbnufft(image, ktraj, smap, im_size, device=torch.device(
        'cpu'), sparse_mats_flag=False)
    profile_torchkbnufft(image, ktraj, smap, im_size, device=torch.device(
        'cpu'), sparse_mats_flag=True)
    profile_torchkbnufft(image, ktraj, smap, im_size, device=torch.device(
        'cuda'), sparse_mats_flag=False)
    profile_torchkbnufft(image, ktraj, smap, im_size, device=torch.device(
        'cuda'), sparse_mats_flag=True)


if __name__ == '__main__':
    run_all_profiles()
