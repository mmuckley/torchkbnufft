import pickle

import numpy as np
import torch
from skimage.data import camera

import torchkbnufft as tkbn
from torchkbnufft.kbinterp import KbInterpAdjoint, KbInterpForward


def test_interp():
    np.random.seed(15)
    dtype = torch.double
    image = np.array(camera()).astype(np.float)
    im_size = tuple(image.shape)
    image = image + np.random.normal(size=im_size) + 1j * np.random.normal(size=im_size)
    image = image / np.max(np.absolute(image))
    image = np.fft.fftn(image)

    # create a k-space trajectory and plot it
    spokelength = image.shape[-1] * 2
    grid_size = (spokelength, spokelength)
    nspokes = 405

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

    image = torch.tensor(image).to(dtype=torch.complex128).unsqueeze(0).unsqueeze(0)
    ktraj = torch.tensor(ktraj, dtype=dtype)

    kb_op = KbInterpForward(im_size=im_size, grid_size=im_size, dtype=torch.complex128)

    kb_op_adj = KbInterpAdjoint(
        im_size=im_size, grid_size=im_size, dtype=torch.complex128
    )

    spmat = tkbn.build_tensor_spmatrix(
        ktraj,
        kb_op.numpoints.numpy(),
        im_size,
        im_size,
        kb_op.n_shift.numpy(),
        kb_op.order.numpy(),
        kb_op.alpha.numpy()
    )

    kdat = kb_op(image, ktraj)
    kdat2 = kb_op(image, ktraj, spmat)

    im_2 = kb_op_adj(kdat, ktraj)
    im_3 = kb_op_adj(kdat, ktraj, spmat)

    with open("result.pkl", "wb") as f:
        pickle.dump((kdat, im_2), f)


if __name__ == "__main__":
    test_interp()
