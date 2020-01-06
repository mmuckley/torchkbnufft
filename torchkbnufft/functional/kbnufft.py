from torch.autograd import Function

from ..nufft.fft_functions import (ifft_and_scale_on_gridded_data,
                                   scale_and_fft_on_image_volume,
                                   fft_filter)
from .kbinterp import AdjKbInterpFunction, KbInterpFunction


class KbNufftFunction(Function):
    @staticmethod
    def forward(ctx, x, om, interpob, interp_mats=None):
        """Apply forward NUFFT.

        This function wraps scale_and_fft_on_image_volume and KbInterpFunction
        for PyTorch autograd.
        """
        x = x.clone()

        # extract interpolation params
        scaling_coef = interpob['scaling_coef']
        grid_size = interpob['grid_size']
        im_size = interpob['im_size']
        norm = interpob['norm']

        x = scale_and_fft_on_image_volume(
            x, scaling_coef, grid_size, im_size, norm)

        y = KbInterpFunction.apply(x, om, interpob, interp_mats)

        ctx.save_for_backward(om)
        ctx.interpob = interpob
        ctx.interp_mats = interp_mats

        return y

    @staticmethod
    def backward(ctx, y):
        """Apply NUFFT adjoint for gradient calculation.

        This function wraps ifft_and_scale_on_gridded_data and AdjKbInterpFunction
        for PyTorch autograd.
        """
        om, = ctx.saved_tensors
        interpob = ctx.interpob
        interp_mats = ctx.interp_mats

        scaling_coef = interpob['scaling_coef']
        grid_size = interpob['grid_size']
        im_size = interpob['im_size']
        norm = interpob['norm']

        interp_mats = ctx.interp_mats

        x = AdjKbInterpFunction.apply(y, om, interpob, interp_mats)

        x = ifft_and_scale_on_gridded_data(
            x, scaling_coef, grid_size, im_size, norm)

        return x, None, None, None


class AdjKbNufftFunction(Function):
    @staticmethod
    def forward(ctx, y, om, interpob, interp_mats=None):
        """Apply NUFFT adjoint.

        This function wraps ifft_and_scale_on_gridded_data and AdjKbInterpFunction
        for PyTorch autograd.
        """
        x = AdjKbInterpFunction.apply(y, om, interpob, interp_mats)

        scaling_coef = interpob['scaling_coef']
        grid_size = interpob['grid_size']
        im_size = interpob['im_size']
        norm = interpob['norm']

        x = ifft_and_scale_on_gridded_data(
            x, scaling_coef, grid_size, im_size, norm)

        ctx.save_for_backward(om)
        ctx.interpob = interpob
        ctx.interp_mats = interp_mats

        return x

    @staticmethod
    def backward(ctx, x):
        """Apply forward NUFFT for gradient calculation.

        This function wraps scale_and_fft_on_image_volume and KbInterpFunction
        for PyTorch autograd.
        """
        x = x.clone()

        om, = ctx.saved_tensors
        interpob = ctx.interpob
        interp_mats = ctx.interp_mats

        scaling_coef = interpob['scaling_coef']
        grid_size = interpob['grid_size']
        im_size = interpob['im_size']
        norm = interpob['norm']

        x = scale_and_fft_on_image_volume(
            x, scaling_coef, grid_size, im_size, norm)

        y = KbInterpFunction.apply(x, om, interpob, interp_mats)

        return y, None, None, None


class ToepNufftFunction(Function):
    @staticmethod
    def forward(ctx, x, kern, norm=None):
        """Apply forward (or adjoint) Toeplitz NUFFT.

        This function wraps fft_filter.
        """
        x = fft_filter(x, kern, norm=norm)

        ctx.save_for_backward(kern)
        ctx.norm = norm

        return x

    @staticmethod
    def backward(ctx, x):
        """Apply adjoint (or forward) Toeplitz NUFFT for gradient calculation.

        This function wraps fft_filter.
        """
        kern, = ctx.saved_tensors
        norm = ctx.norm

        x = fft_filter(x, kern, norm=norm)

        return x, None, None
