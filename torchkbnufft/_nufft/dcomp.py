from typing import Optional, Sequence, Union

import torch
import torchkbnufft.functional as tkbnF
from torch import Tensor

from .utils import init_fn


def calc_density_compensation_function(
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

    This function has optional parameters for initializing a NUFFT object. See
    :py:meth:`~torchkbnufft.KbInterp` for details.

    * :attr:`ktraj` should be of size ``(len(im_size), klength)``,
      where ``klength`` is the length of the k-space trajectory.

    Based on the `method of Pipe
    <https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V>`_.

    This code was contributed by Chaithya G.R.

    Args:
        ktraj: k-space trajectory (in radians/voxel).
        im_size: Size of image with length being the number of dimensions.
        num_iterations: Number of iterations.
        grid_size: Size of grid to use for interpolation, typically 1.25 to 2
            times ``im_size``. Default: ``2 * im_size``
        numpoints: Number of neighbors to use for interpolation in each
            dimension. Default: ``6``
        n_shift: Size for fftshift. Default: ``im_size // 2``.
        table_oversamp: Table oversampling factor.
        kbwidth: Size of Kaiser-Bessel kernel.
        order: Order of Kaiser-Bessel kernel.

    Returns:
        The density compensation coefficients for ``ktraj``.

    Examples:

        >>> data = torch.randn(1, 1, 12) + 1j * torch.randn(1, 1, 12)
        >>> omega = torch.rand(2, 12) * 2 * np.pi - np.pi
        >>> dcomp = tkbn.calculate_density_compensation_function(omega, (8, 8))
        >>> adjkb_ob = tkbn.KbNufftAdjoint(im_size=(8, 8))
        >>> image = adjkb_ob(data * dcomp, omega)
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

        norm_new_sig = torch.abs(new_sig)
        test_sig = test_sig / norm_new_sig

    return test_sig
