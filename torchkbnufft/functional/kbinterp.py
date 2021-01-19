from torch.autograd import Function

from ..nufft.interp import (
    spmat_interp,
    spmat_interp_adjoint,
    table_interp,
    table_interp_adjoint,
)


class KbSpmatInterpForward(Function):
    @staticmethod
    def forward(ctx, image, interp_mats):
        """Apply sparse matrix interpolation.

        This is a wrapper for for PyTorch autograd.
        """
        grid_size = torch.tensor(image.shape, dtype=torch.long, device=image.device)
        output = spmat_interp(image, interp_mats)

        if isinstance(interp_mats, tuple):
            ctx.save_for_backward(interp_mats[0], interp_mats[1], grid_size)
        else:
            ctx.save_for_backward(interp_mats, grid_size)

        return output

    @staticmethod
    def backward(ctx, data):
        """Apply sparse matrix interpolation adjoint for gradient calculation.

        This is a wrapper for for PyTorch autograd.
        """
        if len(ctx.saved_tensors) == 3:
            interp_mats = ctx.saved_tensors[:2]
            grid_size = ctx.saved_tensors[2]
        else:
            (interp_mats, grid_size) = ctx.saved_tensors

        x = spmat_interp_adjoint(data, interp_mats, grid_size)

        return x, None


class KbSpmatInterpAdjoint(Function):
    @staticmethod
    def forward(ctx, data, interp_mats, grid_size):
        """Apply sparse matrix interpolation adjoint.

        This is a wrapper for for PyTorch autograd.
        """
        image = spmat_interp_adjoint(data, interp_mats, grid_size)

        if isinstance(interp_mats, tuple):
            ctx.save_for_backward(interp_mats[0], interp_mats[1])
        else:
            ctx.save_for_backward(interp_mats)

        return image

    @staticmethod
    def backward(ctx, image):
        """Apply sparse matrix interpolation for gradient calculation.

        This is a wrapper for for PyTorch autograd.
        """
        if len(ctx.saved_tensors) == 2:
            interp_mats = ctx.saved_tensors
        else:
            (interp_mats,) = ctx.saved_tensors

        y = spmat_interp(image, interp_mats)

        return y, None, None


class KbTableInterpForward(Function):
    @staticmethod
    def forward(ctx, image, omega, tables, n_shift, numpoints, table_oversamp, offsets):
        """Apply table interpolation.

        This is a wrapper for for PyTorch autograd.
        """
        if image.is_complex():
            grid_size = torch.tensor(image.shape[2:], device=image.device)
        else:
            grid_size = torch.tensor(image.shape[2:-1], device=image.device)

        output = table_interp(
            image=image,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
        )

        ctx.save_for_backward(
            omega, n_shift, numpoints, table_oversamp, offsets, grid_size, *tables
        )

        return output

    @staticmethod
    def backward(ctx, data):
        """Apply table interpolation adjoint for gradient calculation.

        This is a wrapper for for PyTorch autograd.
        """
        (
            omega,
            n_shift,
            numpoints,
            table_oversamp,
            offsets,
            grid_size,
        ) = ctx.saved_tensors[:6]
        tables = [table for table in ctx.saved_tensors[6:]]

        image = table_interp_adjoint(
            data=data,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
            grid_size=grid_size,
        )

        return image, None, None, None, None, None, None


class KbTableInterpAdjoint(Function):
    @staticmethod
    def forward(
        ctx, data, omega, tables, n_shift, numpoints, table_oversamp, offsets, grid_size
    ):
        """Apply table interpolation adjoint.

        This is a wrapper for for PyTorch autograd.
        """
        image = table_interp_adjoint(
            data=data,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
            grid_size=grid_size,
        )

        ctx.save_for_backward(
            omega, n_shift, numpoints, table_oversamp, offsets, *tables
        )

        return image

    @staticmethod
    def backward(ctx, image):
        """Apply table interpolation for gradient calculation.

        This is a wrapper for for PyTorch autograd.
        """
        (omega, n_shift, numpoints, table_oversamp, offsets) = ctx.saved_tensors[:5]
        tables = [table for table in ctx.saved_tensors[5:]]

        data = table_interp(
            image=image,
            omega=omega,
            tables=tables,
            n_shift=n_shift,
            numpoints=numpoints,
            table_oversamp=table_oversamp,
            offsets=offsets,
        )

        return data, None, None, None, None, None, None, None
