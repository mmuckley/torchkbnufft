import itertools
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from .._nufft.utils import build_table

DTYPE_MAP = [
    (torch.complex128, torch.float64),
    (torch.complex64, torch.float32),
    (torch.complex32, torch.float16),
]


class KbModule(nn.Module):
    """Parent class for torchkbnufft modules.

    This class inherits from nn.Module. It is mostly used to have a central
    location for all __repr__ calls.
    """

    def __init__(
        self,
        im_size: Sequence[int],
        grid_size: Optional[Sequence[int]] = None,
        numpoints: Union[int, Sequence[int]] = 6,
        n_shift: Optional[Sequence[int]] = None,
        table_oversamp: Union[int, Sequence[int]] = 2 ** 10,
        kbwidth: float = 2.34,
        order: Union[float, Sequence[float]] = 0.0,
        dtype: torch.dtype = None,
    ):
        super().__init__()

        im_size = tuple(im_size)
        if grid_size is None:
            grid_size = tuple([dim * 2 for dim in im_size])
        else:
            grid_size = tuple(grid_size)
        if isinstance(numpoints, int):
            numpoints = tuple([numpoints for _ in range(len(grid_size))])
        else:
            numpoints = tuple(numpoints)
        if n_shift is None:
            n_shift = tuple([dim // 2 for dim in im_size])
        else:
            n_shift = tuple(n_shift)
        if isinstance(table_oversamp, int):
            table_oversamp = tuple(table_oversamp for _ in range(len(grid_size)))
        else:
            table_oversamp = tuple(table_oversamp)
        alpha = tuple(kbwidth * numpoint for numpoint in numpoints)
        if isinstance(order, float):
            order = tuple(order for _ in range(len(grid_size)))
        else:
            order = tuple(order)
        if dtype is None:
            dtype = torch.get_default_dtype()

        # dimension checking
        assert len(grid_size) == len(im_size)
        assert len(n_shift) == len(im_size)
        assert len(numpoints) == len(im_size)
        assert len(alpha) == len(im_size)
        assert len(order) == len(im_size)
        assert len(table_oversamp) == len(im_size)

        tables = build_table(
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            grid_size=grid_size,
            im_size=im_size,
            order=order,
            alpha=alpha,
        )
        assert len(tables) == len(im_size)

        # precompute interpolation offsets
        offset_list = list(
            itertools.product(*[range(numpoint) for numpoint in numpoints])
        )

        if dtype.is_floating_point:
            real_dtype = dtype
            for pair in DTYPE_MAP:
                if pair[1] == real_dtype:
                    complex_dtype = pair[0]
                    break
        elif dtype.is_complex:
            complex_dtype = dtype
            for pair in DTYPE_MAP:
                if pair[0] == complex_dtype:
                    real_dtype = pair[1]
                    break
        else:
            raise TypeError("Unrecognized dtype.")

        # register all variables as tensor buffers
        for i, table in enumerate(tables):
            self.register_buffer(f"table_{i}", table.to(complex_dtype))
        self.register_buffer("im_size", torch.tensor(im_size, dtype=torch.long))
        self.register_buffer("grid_size", torch.tensor(grid_size, dtype=torch.long))
        self.register_buffer("n_shift", torch.tensor(n_shift, dtype=real_dtype))
        self.register_buffer("numpoints", torch.tensor(numpoints, dtype=torch.long))
        self.register_buffer("offsets", torch.tensor(offset_list, dtype=torch.long))
        self.register_buffer(
            "table_oversamp", torch.tensor(table_oversamp, dtype=torch.long)
        )
        self.register_buffer("order", torch.tensor(order, dtype=real_dtype))
        self.register_buffer("alpha", torch.tensor(alpha, dtype=real_dtype))

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
            if t.is_floating_point():
                cur_dtype = real_dtype
            elif t.is_complex():
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
        filter_list = ["interpob", "buffer", "parameters", "hook", "module"]
        tablecheck = False
        out = "\n{}\n".format(self.__class__.__name__)
        out = out + "----------------------------------------\n"
        for attr, value in self.__dict__.items():
            if "table" in attr:
                if not tablecheck:
                    out = out + "   table: {} arrays, lengths: {}\n".format(
                        len(self.table), self.table_oversamp
                    )
                    tablecheck = True
            elif "traj" in attr or "scaling_coef" in attr:
                out = out + "   {}: {} {} array\n".format(
                    attr, value.shape, value.dtype
                )
            elif not any([item in attr for item in filter_list]):
                out = out + "   {}: {}\n".format(attr, value)
        return out
