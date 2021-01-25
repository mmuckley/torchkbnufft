import time

import numpy as np
import torch
from PIL import Image
from skimage.data import camera

import torchkbnufft as tkbn


def profile_torchkbnufft(
    image, ktraj, smap, im_size, device, sparse_mats_flag=False, toep_flag=False
):
    # run double precision for CPU, float for GPU
    # these seem to be present in reference implementations
    if device == torch.device("cpu"):
        complex_dtype = torch.complex128
        real_dtype = torch.double
        if toep_flag:
            num_nuffts = 20
        else:
            num_nuffts = 5
    else:
        complex_dtype = torch.complex64
        real_dtype = torch.float
        if toep_flag:
            num_nuffts = 50
        else:
            num_nuffts = 20
    cpudevice = torch.device("cpu")

    res = ""
    image = image.to(dtype=complex_dtype)
    ktraj = ktraj.to(dtype=real_dtype)
    smap = smap.to(dtype=complex_dtype)
    interp_mats = None

    forw_ob = tkbn.KbNufft(im_size=im_size, dtype=complex_dtype, device=device)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size, dtype=complex_dtype, device=device)

    # precompute toeplitz kernel if using toeplitz
    if toep_flag:
        kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size)
        toep_ob = tkbn.ToepNufft()

    # precompute the sparse interpolation matrices
    if sparse_mats_flag:
        interp_mats = tkbn.calc_tensor_spmatrix(
            ktraj,
            im_size,
        )
        interp_mats = tuple([t.to(device) for t in interp_mats])
    if toep_flag:
        # warm-up computation
        for _ in range(num_nuffts):
            x = toep_ob(
                image.to(device=device),
                kernel.to(device=device),
                smaps=smap.to(device=device),
            ).to(cpudevice)
        # run the speed tests
        if device == torch.device("cuda"):
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_nuffts):
            x = toep_ob(
                image.to(device=device), kernel.to(device=device), smaps=smap.to(device)
            )
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
            max_mem = torch.cuda.max_memory_allocated()
            res += "GPU forward max memory: {} GB, ".format(max_mem / 1e9)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_nuffts
        res += "toeplitz forward/backward average time: {}".format(avg_time)
    else:
        # warm-up computation
        for _ in range(num_nuffts):
            y = forw_ob(
                image.to(device=device),
                ktraj.to(device=device),
                interp_mats,
                smaps=smap.to(device),
            ).to(cpudevice)

        # run the forward speed tests
        if device == torch.device("cuda"):
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_nuffts):
            y = forw_ob(
                image.to(device=device),
                ktraj.to(device=device),
                interp_mats,
                smaps=smap.to(device),
            )
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
            max_mem = torch.cuda.max_memory_allocated()
            res += "GPU forward max memory: {} GB, ".format(max_mem / 1e9)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_nuffts
        res += "forward average time: {}, ".format(avg_time)

        # warm-up computation
        for _ in range(num_nuffts):
            x = adj_ob(
                y.to(device), ktraj.to(device), interp_mats, smaps=smap.to(device)
            )

        # run the adjoint speed tests
        if device == torch.device("cuda"):
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_nuffts):
            x = adj_ob(
                y.to(device), ktraj.to(device), interp_mats, smaps=smap.to(device)
            )
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
            max_mem = torch.cuda.max_memory_allocated()
            res += "GPU adjoint max memory: {} GB, ".format(max_mem / 1e9)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_nuffts
        res += "backward average time: {}".format(avg_time)

    print(res)


def run_all_profiles():
    print("running profiler...")
    spokelengths = [512]
    nspokes = [405]
    ncoils = [15]
    im_sizes = [256]
    batch_sizes = [1]
    devices = [torch.device("cpu")]
    sparse_mat_flags = [False, True]
    toep_flags = [False, True]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    params = [
        (sl, ns, nc, ims, bs, dev, smf, tf)
        for sl in spokelengths
        for ns in nspokes
        for nc in ncoils
        for ims in im_sizes
        for bs in batch_sizes
        for dev in devices
        for smf in sparse_mat_flags
        for tf in toep_flags
    ]

    for (
        spokelength,
        nspoke,
        ncoil,
        im_size,
        batch_size,
        device,
        sparse_mat_flag,
        toep_flag,
    ) in params:
        if sparse_mat_flag and toep_flag:
            continue
        print(
            f"spokelength: {spokelength}, num spokes: {nspoke}, ncoil: {ncoil}, "
            f"batch_size: {batch_size}, device: {device}, "
            f"sparse_mats: {sparse_mat_flag}, toep_mat: {toep_flag}"
        )

        # create an example to run on
        image = np.array(Image.fromarray(camera()).resize((256, 256)))
        image = image.astype(np.complex)
        im_size = image.shape

        image = (
            torch.tensor(image, dtype=torch.complex128)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1, 1)
        )

        # create k-space trajectory
        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(spokelength, nspoke))
        ky = np.zeros(shape=(spokelength, nspoke))
        ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
        for i in range(1, nspoke):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]

        ky = np.transpose(ky)
        kx = np.transpose(kx)

        ktraj = torch.tensor(np.stack((ky.flatten(), kx.flatten()), axis=0))

        smap_sz = (batch_size, ncoil) + im_size
        smap = torch.ones(*smap_sz, dtype=torch.complex128)

        profile_torchkbnufft(
            image,
            ktraj,
            smap,
            im_size,
            device=device,
            sparse_mats_flag=sparse_mat_flag,
            toep_flag=toep_flag,
        )


if __name__ == "__main__":
    run_all_profiles()
