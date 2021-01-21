from typing import List, Optional, Tuple

from torch import Tensor

from .._nufft.fft_functions import fft_and_scale, ifft_and_scale
from .interp import (
    kb_spmat_interp,
    kb_spmat_interp_adjoint,
    kb_table_interp,
    kb_table_interp_adjoint,
)


def kb_spmat_nufft(
    image: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    interp_mats: Tuple[Tensor, Tensor],
    norm: Optional[str] = None,
) -> Tensor:
    image = fft_and_scale(
        image=image,
        scaling_coef=scaling_coef,
        grid_size=grid_size,
        im_size=im_size,
        norm=norm,
    )

    return kb_spmat_interp(
        image=image,
        interp_mats=interp_mats,
    )


def kb_spmat_nufft_adjoint(
    data: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    interp_mats: Tuple[Tensor, Tensor],
    norm: Optional[str] = None,
) -> Tensor:
    data = kb_spmat_interp_adjoint(
        data=data, interp_mats=interp_mats, grid_size=grid_size
    )

    return ifft_and_scale(
        image=data,
        scaling_coef=scaling_coef,
        grid_size=grid_size,
        im_size=im_size,
        norm=norm,
    )


def kb_table_nufft(
    image: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    image = fft_and_scale(
        image=image,
        scaling_coef=scaling_coef,
        grid_size=grid_size,
        im_size=im_size,
        norm=norm,
    )

    return kb_table_interp(
        image=image,
        omega=omega,
        tables=tables,
        n_shift=n_shift,
        numpoints=numpoints,
        table_oversamp=table_oversamp,
        offsets=offsets,
    )


def kb_table_nufft_adjoint(
    data: Tensor,
    scaling_coef: Tensor,
    im_size: Tensor,
    grid_size: Tensor,
    omega: Tensor,
    tables: List[Tensor],
    n_shift: Tensor,
    numpoints: Tensor,
    table_oversamp: Tensor,
    offsets: Tensor,
    norm: Optional[str] = None,
) -> Tensor:
    data = kb_table_interp_adjoint(
        data=data,
        omega=omega,
        tables=tables,
        n_shift=n_shift,
        numpoints=numpoints,
        table_oversamp=table_oversamp,
        offsets=offsets,
        grid_size=grid_size,
    )

    return ifft_and_scale(
        image=data,
        scaling_coef=scaling_coef,
        grid_size=grid_size,
        im_size=im_size,
        norm=norm,
    )
