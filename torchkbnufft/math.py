import torch
from torch import Tensor


def complex_mult(val1: Tensor, val2: Tensor, dim: int = -1) -> Tensor:
    """Complex multiplication, real/imag are in dimension dim.

    Args:
        val1: A tensor to be multiplied.
        val2: A second tensor to be multiplied.
        dim: An integer indicating the complex dimension (for real inputs
            only).

    Returns:
        val1 * val2, where * executes complex multiplication.
    """
    if not val1.dtype == val2.dtype:
        raise ValueError("val1 has different dtype than val2.")

    if torch.is_complex(val1):
        val3 = val1 * val2
    else:
        if not val1.shape[dim] == val2.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

        real_a = val1.select(dim, 0)
        imag_a = val1.select(dim, 1)
        real_b = val2.select(dim, 0)
        imag_b = val2.select(dim, 1)

        val3 = torch.stack(
            (real_a * real_b - imag_a * imag_b, imag_a * real_b + real_a * imag_b), dim
        )

    return val3


def conj_complex_mult(val1: Tensor, val2: Tensor, dim: int = -1) -> Tensor:
    """Complex multiplication, real/imag are in dimension dim.

    Args:
        val1: A tensor to be multiplied.
        val2: A second tensor to be conjugated then multiplied.
        dim: An integer indicating the complex dimension (for real inputs
            only).

    Returns:
        val3 = val1 * conj(val2), where * executes complex multiplication.
    """
    if not val1.dtype == val2.dtype:
        raise ValueError("val1 has different dtype than val2.")

    if torch.is_complex(val1):
        val3 = val1 * val2.conj()
    else:
        if not val1.shape[dim] == val2.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

        real_a = val1.select(dim, 0)
        imag_a = val1.select(dim, 1)
        real_b = val2.select(dim, 0)
        imag_b = val2.select(dim, 1)

        val3 = torch.stack(
            (real_a * real_b + imag_a * imag_b, imag_a * real_b - real_a * imag_b), dim
        )

    return val3


def imag_exp(val: Tensor, dim: int = -1, return_complex: bool = True) -> Tensor:
    """Imaginary exponential, exp(ia), returns real/imag separate if real.

    Args:
        val: A tensor to be exponentiated.
        dim: An integer indicating the complex dimension of the output (for
            real outputs only).

    Returns:
        val2 = exp(i*val), where i is sqrt(-1).
    """
    val2 = torch.stack((torch.cos(val), torch.sin(val)), -1)
    if return_complex:
        val2 = torch.view_as_complex(val2)

    return val2


def inner_product(val1: Tensor, val2: Tensor, dim: int = -1) -> Tensor:
    """Complex inner product.

    Args:
        val1: A tensor for the inner product.
        val2: A second tensor for the inner product.
        dim: An integer indicating the complex dimension (for real inputs
            only).

    Returns:
        The complex inner product of val1 and val2.
    """
    if not val1.dtype == val2.dtype:
        raise ValueError("val1 has different dtype than val2.")

    if not torch.is_complex(val1):
        if not val1.shape[dim] == val2.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

    inprod = conj_complex_mult(val2, val1, dim=dim)

    if not torch.is_complex(val1):
        inprod = torch.cat((inprod.select(dim, 0).sum(), inprod.select(dim, 1).sum()))
    else:
        inprod = torch.sum(inprod)

    return inprod


def absolute(val: Tensor, dim: int = -1) -> Tensor:
    """Complex absolute value.

    Args:
        val: A tensor to have its absolute value computed.
        dim: An integer indicating the complex dimension (for real inputs
            only).

    Returns:
        tensor: The absolute value of t.
    """
    if torch.is_complex(val):
        abs_val = torch.abs(val)
    else:
        if not val.shape[dim] == 2:
            raise ValueError("Real input does not have dimension size 2 at dim.")

        abs_val = torch.sqrt(
            val.select(dim, 0) ** 2 + val.select(dim, 1) ** 2
        ).unsqueeze(dim)

    return abs_val


def complex_sign(val: Tensor, dim: int = -1) -> Tensor:
    """Complex sign function value.

    Args:
        val: A tensor to have its complex sign computed.
        dim: An integer indicating the complex dimension (for real inputs
            only).

    Returns:
        The complex sign of val.
    """
    is_complex = False
    if torch.is_complex(val):
        is_complex = True
        val = torch.view_as_real(val)
        dim = -1
    elif not val.shape[dim] == 2:
        raise ValueError("Real input does not have dimension size 2 at dim.")

    sign_val = torch.atan2(val.select(dim, 1), val.select(dim, 0))
    sign_val = imag_exp(sign_val, dim=dim, return_complex=is_complex)

    return sign_val
