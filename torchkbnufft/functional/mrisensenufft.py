from torch.autograd import Function

from ..mri.sensenufft_functions import (
    coilpack_sense_backward,
    coilpack_sense_forward,
    sense_backward,
    sense_forward,
    sense_toeplitz,
)
from .kbnufft import AdjKbNufftFunction, KbNufftFunction


class MriSenseNufftFunction(Function):
    @staticmethod
    def forward(ctx, x, smap, om, interpob, interp_mats=None):
        """Apply forward SENSE operation into k-space.

        This function wraps sense_forward and coilpack_sense_forward for PyTorch
        autograd.
        """
        if interpob["coilpack"]:
            y = coilpack_sense_forward(x, smap, om, interpob, interp_mats)
        else:
            y = sense_forward(x, smap, om, interpob, interp_mats)

        ctx.save_for_backward(smap, om)
        ctx.interpob = interpob
        ctx.interp_mats = interp_mats

        return y

    @staticmethod
    def backward(ctx, y):
        """Apply adjoint SENSE operation into image space.

        This function wraps sense_backward and coilpack_sense_backward for PyTorch
        autograd.
        """
        smap, om = ctx.saved_tensors
        interpob = ctx.interpob
        interp_mats = ctx.interp_mats

        if interpob["coilpack"]:
            x = coilpack_sense_backward(y, smap, om, interpob, interp_mats)
        else:
            x = sense_backward(y, smap, om, interpob, interp_mats)

        return x, None, None, None, None


class AdjMriSenseNufftFunction(Function):
    @staticmethod
    def forward(ctx, y, smap, om, interpob, interp_mats=None):
        """Apply adjoint SENSE operation into image space.

        This function wraps sense_backward and coilpack_sense_backward for PyTorch
        autograd.
        """
        if interpob["coilpack"]:
            x = coilpack_sense_backward(y, smap, om, interpob, interp_mats)
        else:
            x = sense_backward(y, smap, om, interpob, interp_mats)

        ctx.save_for_backward(smap, om)
        ctx.interpob = interpob
        ctx.interp_mats = interp_mats

        return x

    @staticmethod
    def backward(ctx, x):
        """Apply forward SENSE operation into k-space.

        This function wraps sense_forward and coilpack_sense_forward for PyTorch
        autograd.
        """
        smap, om = ctx.saved_tensors
        interpob = ctx.interpob
        interp_mats = ctx.interp_mats

        if interpob["coilpack"]:
            y = coilpack_sense_forward(x, smap, om, interpob, interp_mats)
        else:
            y = sense_forward(x, smap, om, interpob, interp_mats)

        return y, None, None, None, None


class ToepSenseNufftFunction(Function):
    @staticmethod
    def forward(ctx, x, smap, kern, norm=None):
        """Apply forward (or adjoint) ToepSense NUFFT.

        This function wraps sense_toeplitz.
        """
        x = sense_toeplitz(x, smap, kern, norm=norm)

        ctx.save_for_backward(smap, kern)
        ctx.norm = norm

        return x

    @staticmethod
    def backward(ctx, x):
        """Apply adjoint (or forward) ToepSense NUFFT for gradient calculation.

        This function wraps sense_toeplitz.
        """
        (
            smap,
            kern,
        ) = ctx.saved_tensors
        norm = ctx.norm

        x = sense_toeplitz(x, smap, kern, norm=norm)

        return x, None, None, None
