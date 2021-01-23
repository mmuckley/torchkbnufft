from typing import Optional, Sequence, Union

import torch
import torchkbnufft.functional as tkbnF
from torch import Tensor

from .utils import init_fn


def calculate_density_compensation_function(
    ktraj: Tensor,
    im_size: Sequence[int],
    num_iterations: int = 10,
    grid_size: Optional[Sequence[int]] = None,
    numpoints: Union[int, Sequence[int]] = 6,
    n_shift: Optional[Sequence[int]] = None,
    table_oversamp: Union[int, Sequence[int]] = 2 ** 10,
    kbwidth: float = 2.34,
    order: Union[float, Sequence[float]] = 0.0,
) -> Tensor:
    """Numerical density compensation estimation for any trajectory.

    Based on the method of Pipe:

    Pipe JG, Menon P. Sampling density compensation in MRI: rationale and an
    iterative numerical solution. MRM. 1999 Jan;41(1):179-86.

    This code was contributed by Chaithya G.R. and Z. Ramzi.

    Args:
        ktraj: The k-space trajectory in radians/voxel dimension ``(d, m)``.
            ``d`` is the number of spatial dimensions, and m is the length of
            the trajectory.
        im_size: Size of image.
        num_iterations: Number of iterations.
        grid_size; Optional: Size of grid to use for interpolation, typically
            1.25 to 2 times ``im_size``.
        numpoints: Number of neighbors to use for interpolation.
        n_shift; Optional: Size for fftshift, usually ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        offsets: A list of offsets, looping over all possible combinations of
            ``numpoints``.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.

    Returns:
        The density compensation coefficients for ``ktraj``.
    """
    device = ktraj.device

    # init nufft variables
    (
        tables,
        _,
        grid_size_t,
        n_shift_t,
        numpoints_t,
        offsets_t,
        table_oversamp_t,
        _,
        _,
    ) = init_fn(
        im_size=im_size,
        grid_size=grid_size,
        numpoints=numpoints,
        n_shift=n_shift,
        table_oversamp=table_oversamp,
        kbwidth=kbwidth,
        order=order,
        dtype=ktraj.dtype,
        device=device,
    )

    tables = [table.to(device) for table in tables]
    n_shift_t = n_shift_t.to(device)
    numpoints_t = numpoints_t.to(device)
    table_oversamp_t = table_oversamp_t.to(device)
    offsets_t = offsets_t.to(device)
    grid_size_t = grid_size_t.to(device)

    test_sig = torch.ones([1, 1, ktraj.shape[-1]], dtype=tables[0].dtype, device=device)
    for _ in range(num_iterations):
        new_sig = tkbnF.kb_table_interp(
            tkbnF.kb_table_interp_adjoint(
                data=test_sig,
                omega=ktraj,
                tables=tables,
                n_shift=n_shift_t,
                numpoints=numpoints_t,
                table_oversamp=table_oversamp_t,
                offsets=offsets_t,
                grid_size=grid_size_t,
            ),
            omega=ktraj,
            tables=tables,
            n_shift=n_shift_t,
            numpoints=numpoints_t,
            table_oversamp=table_oversamp_t,
            offsets=offsets_t,
        )

        # Basically we are doing abs here, do we have utils for this?
        norm_new_sig = torch.abs(new_sig)
        test_sig = test_sig / norm_new_sig

    return test_sig
