from PIL import Image
import time
import torch
import torchkbnufft as tkbn
from skimage.data import camera
import numpy as np


def run_batched_test():
    # create an example to run on
    spokelength = 512
    nspoke = 100
    batch_size = 1

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

    ktraj = torch.tensor(np.stack((ky, kx), axis=1))
    image = image.repeat(100, 1, 1, 1)

    smap_sz = (100, 15) + im_size
    smap = torch.ones(*smap_sz, dtype=torch.complex128)

    forw_ob = tkbn.KbNufft(im_size=im_size).to(image)
    adj_ob = tkbn.KbNufftAdjoint(im_size=im_size).to(image)

    num_nuffts = 5

    print("forward runs...")

    # warmup
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            data = forw_ob(image[i].unsqueeze(0), ktraj[i], smaps=smap)

    # speed tests
    forloop_time_start = time.perf_counter()
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            data = forw_ob(image[i].unsqueeze(0), ktraj[i], smaps=smap)
    forloop_time_end = time.perf_counter()
    for _ in range(num_nuffts):
        data = forw_ob(image, ktraj, smaps=smap)
    batch_time_end = time.perf_counter()

    print(
        f"cpu forloop time: {(forloop_time_end-forloop_time_start) / num_nuffts}, "
        f"cpu batch time: {(batch_time_end-forloop_time_end) / num_nuffts}"
    )

    print("adjoint runs...")
    # warmup
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            image = adj_ob(data[i].unsqueeze(0), ktraj[i], smaps=smap)

    # speed tests
    forloop_time_start = time.perf_counter()
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            image = adj_ob(data[i].unsqueeze(0), ktraj[i], smaps=smap)
    forloop_time_end = time.perf_counter()
    for _ in range(num_nuffts):
        image = adj_ob(data, ktraj, smaps=smap)
    batch_time_end = time.perf_counter()

    print(
        f"cpu forloop time: {(forloop_time_end-forloop_time_start) / num_nuffts}, "
        f"cpu batch time: {(batch_time_end-forloop_time_end) / num_nuffts}"
    )

    print("switching to GPU")
    num_nuffts = 15
    image = image.to(device=torch.device("cuda"), dtype=torch.complex64)
    data = data.to(image)
    forw_ob = forw_ob.to(image)
    adj_ob = adj_ob.to(image)
    ktraj = ktraj.to(device=torch.device("cuda"), dtype=torch.float32)

    # warmup
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            data = forw_ob(image[i].unsqueeze(0), ktraj[i], smaps=smap)

    # speed tests
    forloop_time_start = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            data = forw_ob(image[i].unsqueeze(0), ktraj[i], smaps=smap)
    torch.cuda.synchronize()
    forloop_time_end = time.perf_counter()
    for _ in range(num_nuffts):
        data = forw_ob(image, ktraj, smaps=smap)
    torch.cuda.synchronize()
    batch_time_end = time.perf_counter()

    print(
        f"gpu forloop time: {(forloop_time_end-forloop_time_start) / num_nuffts}, "
        f"gpu batch time: {(batch_time_end-forloop_time_end) / num_nuffts}"
    )

    print("adjoint runs...")
    # warmup
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            image = adj_ob(data[i].unsqueeze(0), ktraj[i], smaps=smap)

    # speed tests
    forloop_time_start = time.perf_counter()
    torch.cuda.synchronize()
    for _ in range(num_nuffts):
        for i in range(len(ktraj)):
            image = adj_ob(data[i].unsqueeze(0), ktraj[i], smaps=smap)
    torch.cuda.synchronize()
    forloop_time_end = time.perf_counter()
    for _ in range(num_nuffts):
        image = adj_ob(data, ktraj, smaps=smap)
    torch.cuda.synchronize()
    batch_time_end = time.perf_counter()

    print(
        f"gpu forloop time: {(forloop_time_end-forloop_time_start) / num_nuffts}, "
        f"gpu batch time: {(batch_time_end-forloop_time_end) / num_nuffts}"
    )


if __name__ == "__main__":
    run_batched_test()
