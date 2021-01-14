import torch


def complex_mult(a, b, dim=0):
    """Complex multiplication, real/imag are in dimension dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        b (tensor): A tensor where dimension dim is the complex dimension.
        dim (int): An integer indicating the complex dimension.

    Returns:
        tensor: a * b, where * executes complex multiplication.
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    real_a = a.select(dim, 0)
    imag_a = a.select(dim, 1)
    real_b = b.select(dim, 0)
    imag_b = b.select(dim, 1)

    c = torch.stack(
        (real_a * real_b - imag_a * imag_b, imag_a * real_b + real_a * imag_b), dim
    )

    return c


def conj_complex_mult(a, b, dim=0):
    """Complex multiplication, real/imag are in dimension dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        b (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: c = a * conj(b), where * executes complex multiplication.
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    real_a = a.select(dim, 0)
    imag_a = a.select(dim, 1)
    real_b = b.select(dim, 0)
    imag_b = b.select(dim, 1)

    c = torch.stack(
        (real_a * real_b + imag_a * imag_b, imag_a * real_b - real_a * imag_b), dim
    )

    return c


def imag_exp(a, dim=0):
    """Imaginary exponential, exp(ia), returns real/imag separate in dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension of
            the output.

    Returns:
        tensor: c = exp(i*a), where i is sqrt(-1).
    """
    c = torch.stack((torch.cos(a), torch.sin(a)), dim)

    return c


def inner_product(a, b, dim=0):
    """Complex inner product, complex dimension is dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        b (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: The complex inner product of a and b of size 2 (real, imag).
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    inprod = conj_complex_mult(b, a, dim=dim)

    real_inprod = inprod.select(dim, 0).sum()
    imag_inprod = inprod.select(dim, 1).sum()

    return torch.cat((real_inprod.view(1), imag_inprod.view(1)))


def absolute(t, dim=0):
    """Complex absolute value, complex dimension is dim.

    Args:
        t (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: The absolute value of t.
    """
    assert t.shape[dim] == 2

    abst = torch.sqrt(t.select(dim, 0) ** 2 + t.select(dim, 1) ** 2).unsqueeze(dim)

    return abst


def complex_sign(t, dim=0):
    """Complex sign function value, complex dimension is dim.

    Args:
        t (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: The complex sign of t.
    """
    assert t.shape[dim] == 2

    signt = torch.atan2(t.select(dim, 1), t.select(dim, 0))
    signt = imag_exp(signt, dim=dim)

    return signt
