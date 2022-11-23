from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .._nufft.utils import DTYPE_MAP, init_fn


class KbModule(nn.Module):
    """Parent class for torchkbnufft modules.

    This class handles initialization of NUFFT precomputations and registers
    the resulting tensors as buffers.
    """

    def __init__(
        self,
        im_size: Sequence[int],
        grid_size: Optional[Sequence[int]] = None,
        numpoints: Union[int, Sequence[int]] = 6,
        n_shift: Optional[Sequence[int]] = None,
        table_oversamp: Union[int, Sequence[int]] = 2**10,
        kbwidth: float = 2.34,
        order: Union[float, Sequence[float]] = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        (
            tables,
            im_size_t,
            grid_size_t,
            n_shift_t,
            numpoints_t,
            offsets_t,
            table_oversamp_t,
            order_t,
            alpha_t,
        ) = init_fn(
            im_size=im_size,
            grid_size=grid_size,
            numpoints=numpoints,
            n_shift=n_shift,
            table_oversamp=table_oversamp,
            kbwidth=kbwidth,
            order=order,
            dtype=dtype,
            device=device,
        )

        # register all variables as tensor buffers
        for i, table in enumerate(tables):
            self.register_buffer(f"table_{i}", table)
        self.register_buffer("im_size", im_size_t)
        self.register_buffer("grid_size", grid_size_t)
        self.register_buffer("n_shift", n_shift_t)
        self.register_buffer("numpoints", numpoints_t)
        self.register_buffer("offsets", offsets_t)
        self.register_buffer("table_oversamp", table_oversamp_t)
        self.register_buffer("order", order_t)
        self.register_buffer("alpha", alpha_t)

    def to(self, *args, **kwargs):
        """Rewrite nn.Module.to to support complex floats."""

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError(
                    "KbModule.to only accepts floating point or complex "
                    "dtypes, but got desired dtype={}".format(dtype)
                )

            if dtype.is_complex:
                complex_dtype = dtype
                for pair in DTYPE_MAP:
                    if pair[0] == complex_dtype:
                        real_dtype = pair[1]
                        break
            elif dtype.is_floating_point:
                real_dtype = dtype
                for pair in DTYPE_MAP:
                    if pair[1] == real_dtype:
                        complex_dtype = pair[0]
                        break
            else:
                raise TypeError("Unrecognized type.")

        def convert(t):
            if t.is_floating_point() and dtype is not None:
                cur_dtype = real_dtype
            elif t.is_complex() and dtype is not None:
                cur_dtype = complex_dtype
            else:
                cur_dtype = None

            if convert_to_format is not None and t.dim() == 4:
                return t.to(
                    device, cur_dtype, non_blocking, memory_format=convert_to_format
                )

            return t.to(device, cur_dtype, non_blocking)

        return self._apply(convert)

    def __repr__(self):
        out = "\n{}\n".format(self.__class__.__name__)
        out = out + "----------------------------------------\n"
        out = out + "buffers\n"
        for buf, val in self.__dict__["_buffers"].items():
            out = out + f"\ttensor: {buf}, shape: {tuple(val.shape)}\n"
        return out
