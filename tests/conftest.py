import numpy as np
import torch
import torchkbnufft as tkbn


def test_params():
    batch_size = 5
    ncoil = 4
    klength = 112

    im_size_2d = (33, 24)
    im_size_3d = (11, 33, 24)

    numpoints_2d = (4, 6)
    numpoints_3d = (2, 4, 6)

    return (
        batch_size,
        ncoil,
        klength,
        im_size_2d,
        im_size_3d,
        numpoints_2d,
        numpoints_3d,
    )


def create_input_plus_noise(shape, is_complex):
    x = np.arange(np.product(shape)).reshape(shape)
    x = torch.tensor(x, dtype=torch.get_default_dtype())

    if is_complex:
        x = x + torch.randn(size=x.shape) + 1j * torch.randn(size=x.shape)
    else:
        x = x + torch.randn(size=x.shape)

    return x


def nufft_adjoint_test(image, kdata, ktraj, forw_ob, adj_ob, spmat):
    image_forw = forw_ob(image, ktraj)
    kdata_adj = adj_ob(kdata, ktraj)

    assert torch.allclose(
        tkbn.inner_product(image_forw, kdata), tkbn.inner_product(image, kdata_adj)
    )

    image_forw = forw_ob(image, ktraj, spmat)
    kdata_adj = adj_ob(kdata, ktraj, spmat)

    assert torch.allclose(
        tkbn.inner_product(image_forw, kdata), tkbn.inner_product(image, kdata_adj)
    )


def nufft_autograd_test(image, kdata, ktraj, forw_ob, adj_ob, spmat):
    image.requires_grad = True
    kdata.requires_grad = True
    image_forw = forw_ob(image, ktraj)
    kdata_adj = adj_ob(kdata, ktraj)

    (torch.abs(image_forw) ** 2 / 2).sum().backward()
    (torch.abs(kdata_adj) ** 2 / 2).sum().backward()
    autograd_forw = image.grad.clone()
    autograd_adj = kdata.grad.clone()
    grad_forw_est = adj_ob(image_forw.detach(), ktraj)
    grad_adj_est = forw_ob(kdata_adj.detach(), ktraj)

    assert torch.allclose(autograd_forw, grad_forw_est)
    assert torch.allclose(autograd_adj, grad_adj_est)

    image.grad = torch.zeros_like(image.grad)
    kdata.grad = torch.zeros_like(kdata.grad)
    image.requires_grad = True
    kdata.requires_grad = True
    image_forw = forw_ob(image, ktraj, spmat)
    kdata_adj = adj_ob(kdata, ktraj, spmat)

    (torch.abs(image_forw) ** 2 / 2).sum().backward()
    (torch.abs(kdata_adj) ** 2 / 2).sum().backward()
    autograd_forw = image.grad.clone()
    autograd_adj = kdata.grad.clone()
    grad_forw_est = adj_ob(image_forw.detach(), ktraj, spmat)
    grad_adj_est = forw_ob(kdata_adj.detach(), ktraj, spmat)

    assert torch.allclose(autograd_forw, grad_forw_est)
    assert torch.allclose(autograd_adj, grad_adj_est)


def create_ktraj(ndims, klength):
    return torch.rand(size=(ndims, klength)) * 2 * np.pi - np.pi
