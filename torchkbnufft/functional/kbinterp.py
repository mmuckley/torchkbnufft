from torch.autograd import Function

from ..nufft.interp_functions import adjkbinterp, kbinterp


class KbInterpFunction(Function):
    @staticmethod
    def forward(ctx, x, om, interpob, interp_mats=None):
        """Apply table interpolation.

        This is a wrapper for nufft.interp_functions.kbinterp for PyTorch autograd.
        """
        y = kbinterp(x, om, interpob, interp_mats)

        ctx.save_for_backward(om)
        ctx.interpob = interpob
        ctx.interp_mats = interp_mats

        return y

    @staticmethod
    def backward(ctx, y):
        """Apply table interpolation adjoint for gradient calculation.

        This is a wrapper for nufft.interp_functions.adjkbinterp for PyTorch autograd.
        """
        om, = ctx.saved_tensors
        interpob = ctx.interpob
        interp_mats = ctx.interp_mats

        x = adjkbinterp(y, om, interpob, interp_mats)

        return x, None, None, None


class AdjKbInterpFunction(Function):
    @staticmethod
    def forward(ctx, y, om, interpob, interp_mats=None):
        """Apply table interpolation adjoint.

        This is a wrapper for nufft.interp_functions.adjkbinterp for PyTorch autograd.
        """
        x = adjkbinterp(y, om, interpob, interp_mats)

        ctx.save_for_backward(om)
        ctx.interpob = interpob
        ctx.interp_mats = interp_mats

        return x

    @staticmethod
    def backward(ctx, x):
        """Apply table interpolation for gradient calculation.

        This is a wrapper for nufft.interp_functions.kbinterp for PyTorch autograd.
        """
        om, = ctx.saved_tensors
        interpob = ctx.interpob
        interp_mats = ctx.interp_mats

        y = kbinterp(x, om, interpob, interp_mats)

        return y, None, None, None
