from typing import Optional, Sequence, Tuple, Union

import torch
import torchkbnufft.functional as tkbnF
from torch import Tensor

from .._nufft.utils import compute_scaling_coefs
from ._kbmodule import KbModule


class KbNufftModule(KbModule):
    """Parent class for KbNufft classes.

    See subclasses for an explanation of inputs.
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
        device: torch.device = None,
    ):
        super().__init__(
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

        scaling_coef = compute_scaling_coefs(
            im_size=self.im_size.tolist(),  # type: ignore
            grid_size=self.grid_size.tolist(),  # type: ignore
            numpoints=self.numpoints.tolist(),  # type: ignore
            alpha=self.alpha.tolist(),  # type: ignore
            order=self.order.tolist(),  # type: ignore
        )

        self.register_buffer(
            "scaling_coef", scaling_coef.to(self.table_0.dtype)  # type: ignore
        )


class KbNufft(KbNufftModule):
    r"""Non-uniform FFT PyTorch module.

    This object applies the FFT and interpolates a grid of Fourier data to
    off-grid locations using a Kaiser-Bessel kernel. Consider an off-grid
    signal :math:`Y` to be estimated from an image-domain signal :math:`X`,
    then this layer applies:

    .. math::
        X_k = \sum_{n=0}^{N-1} s_n x_n e^{-i \gamma k n}
    .. math::
        Y(\omega_m) = \sum_{j=1}^J X_{\{k_m+j\}_K} u^*_j(\omega_m)

    In the first step, an image-domain signal :math:`x_n` is converted to a
    gridded, oversampled frequency-domain signal, :math:`X_k`. The scaling
    coefficeints :math:`s_n` are multiplied to precompensate for NUFFT
    interpolation errors.

    In the second step, :math:`u`, the Kaiser-Bessel kernel, is used to
    estimate :math:`X_k` at off-grid frequency locations :math:`\omega`.
    :math:`k_m` is the index of the nearest sample of :math:`X` to frequency
    location :math:`\omega_m`, and :math:`J` is the number of nearest neighbors
    to use from the Kaiser-Bessel kernel. For a detailed description see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://ieeexplore.ieee.org/abstract/document/1166689>`_.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image.

    * :attr:`grid_size` is the size of the grid after forward FFT. To reduce
      errors, NUFFT operations are done on an oversampled grid to reduce
      interpolation distances. This will typically be 1.25 to 2 times
      :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors of the kernel to use
      for interpolation.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size (tuple): Size of image with length being the number of
            dimensions.
        grid_size (tuple, optional): Size of grid to use for interpolation,
            typically 1.25 to 2 times ``im_size``. Default: ``2 * im_size``
        numpoints (int or tuple, optional): Number of neighbors to use for
            interpolation in each dimension. Default: ``6``
        n_shift (tuple, optional): Size for fftshift. Default:
            ``im_size // 2``.
        table_oversamp (int or tuple, optional): Table oversampling factor.
            Default: ``2 ** 10``
        kbwidth (float, optional): Size of Kaiser-Bessel kernel. Default:
            ``2.34``
        order (float or tuple, optional): Order of Kaiser-Bessel kernel.
            Default: ``0``
        dtype (torch.dtype, optional): Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device (torch.device, optional): Which device to create tensors on.
            Default: ``torch.device('cpu')``
    """

    def forward(
        self,
        image: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        smaps: Optional[Tensor] = None,
        norm: Optional[str] = None,
    ) -> Tensor:
        """Apply FFT and interpolate from gridded data to scattered data.

        Input tensors should be of shape ``(N, C) + im_size``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size ``(len(im_size), klength)``,
        where ``klength`` is the length of the k-space trajectory.

        If your tensors are real, ensure that 2 is the size of the last
        dimension.

        Args:
            image (Tensor): Object to be NUFFT'd.
            omega (Tensor): k-space trajectory (in radians/voxel).
            interp_mats (tuple, optional): 2-tuple of real, imaginary spars
                matrices to use for sparse matrix NUFFT interpolation
                (overrides default table interpolation).
            smaps (Tensor, optional): Sensitivity maps. If input, these will be
                multiplied before the forward NUFFT.
            norm (string, optional): Whether to apply normalization with the
                FFT operation. Options are ``"ortho"` or ``None``. Default:
                ``None``

        Returns:
            Tensor: ``image`` calculated at Fourier frequencies specified by
            ``omega``.
        """
        if smaps is not None:
            if not smaps.dtype == image.dtype:
                raise TypeError("image dtype does not match smaps dtype.")

        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError("For real inputs, last dimension must be size 2.")

                smaps = torch.view_as_complex(smaps)

            is_complex = False
            image = torch.view_as_complex(image)

        if smaps is not None:
            image = image * smaps

        if interp_mats is not None:
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)

            output = tkbnF.kb_spmat_nufft(
                image=image,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                interp_mats=interp_mats,
                norm=norm,
            )
        else:
            tables = []
            for i in range(len(self.im_size)):  # type: ignore
                tables.append(getattr(self, f"table_{i}"))

            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)

            output = tkbnF.kb_table_nufft(
                image=image,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                omega=omega,
                tables=tables,
                n_shift=self.n_shift,
                numpoints=self.numpoints,
                table_oversamp=self.table_oversamp,
                offsets=self.offsets.to(torch.long),
                norm=norm,
            )

        if not is_complex:
            output = torch.view_as_real(output)

        return output


class KbNufftAdjoint(KbNufftModule):
    r"""Non-uniform FFT adjoint PyTorch module.

    This object interpolates off-grid Fourier data to on-grid locations
    using a Kaiser-Bessel kernel prior to inverse DFT. Consider an image-domain
    signal :math:`X` to be estimated from an off-grid Fourier signal :math:`Y`,
    then this layer applies:

    .. math::
        X_{\{k_m+j\}_K} = \sum_{j=1}^J Y(\omega_m) u_j(\omega_m)
    .. math::
        x_n = s_n^* \sum_{k=0}^{K-1} X_k e^{i \gamma k n}

    In the first step, :math:`u`, the Kaiser-Bessel kernel, is used to
    estimate :math:`Y` at on-grid frequency by using locations :math:`\omega`.
    :math:`k_m` is the index of the nearest sample of :math:`X` to frequency
    location :math:`\omega_m`, and :math:`J` is the number of nearest neighbors
    to use from the Kaiser-Bessel kernel.

    In the second step, an image-domain signal :math:`x_n` is estimated from a
    gridded, oversampled frequency-domain signal, :math:`X_k` by applying the
    inverse FFT, after which the complex conjugate scaling coefficients
    :math:`s_n` are multiplied. For a detailed description see
    `Nonuniform fast Fourier transforms using min-max interpolation
    (JA Fessler and BP Sutton)
    <https://ieeexplore.ieee.org/abstract/document/1166689>`_.

    Note:

        This function is not the inverse of ``KbNufft``, is is the adjoint.

    When called, the parameters of this class define properties of the kernel
    and how the interpolation is applied.

    * :attr:`im_size` is the size of the base image.

    * :attr:`grid_size` is the size of the grid after adjoint interpolation. To
      reduce errors, NUFFT operations are done on an oversampled grid to reduce
      interpolation distances. This will typically be 1.25 to 2 times
      :attr:`im_size`.

    * :attr:`numpoints` is the number of nearest neighbors of the kernel to use
      for interpolation.

    * :attr:`n_shift` is the FFT shift distance, typically
      :attr:`im_size // 2`.

    Args:
        im_size (tuple): Size of image with length being the number of
            dimensions.
        grid_size (tuple, optional): Size of grid to use for interpolation,
            typically 1.25 to 2 times ``im_size``. Default: ``2 * im_size``
        numpoints (int or tuple, optional): Number of neighbors to use for
            interpolation in each dimension. Default: ``6``
        n_shift (tuple, optional): Size for fftshift. Default:
            ``im_size // 2``.
        table_oversamp (int or tuple, optional): Table oversampling factor.
            Default: ``2 ** 10``
        kbwidth (float, optional): Size of Kaiser-Bessel kernel. Default:
            ``2.34``
        order (float or tuple, optional): Order of Kaiser-Bessel kernel.
            Default: ``0``
        dtype (torch.dtype, optional): Data type for tensor buffers. Default:
            ``torch.get_default_dtype()``
        device (torch.device, optional): Which device to create tensors on.
            Default: ``torch.device('cpu')``
    """

    def forward(
        self,
        data: Tensor,
        omega: Tensor,
        interp_mats: Optional[Tuple[Tensor, Tensor]] = None,
        smaps: Optional[Tensor] = None,
        norm: Optional[str] = None,
    ) -> Tensor:
        """Interpolate from scattered data to gridded data and then iFFT.

        Input tensors should be of shape ``(N, C) + klength``, where ``N`` is
        the batch size and ``C`` is the number of sensitivity coils. ``omega``,
        the k-space trajectory, should be of size ``(len(im_size), klength)``,
        where ``klength`` is the length of the k-space trajectory.

        If your tensors are real, ensure that 2 is the size of the last
        dimension.

        Args:
            data (Tensor): Data to be gridded and then inverse FFT'd.
            omega (Tensor): k-space trajectory (in radians/voxel).
            interp_mats (tuple, optional) 2-tuple of real, imaginary sparse
                matrices to use for sparse matrix NUFFT interpolation
                (overrides default table interpolation).
            smaps(Tensor, optional): Sensitivity maps. If input, these will be
                multiplied before the forward NUFFT.
            norm (string, optional): Whether to apply normalization with the
                FFT operation. Options are ``"ortho"`` or ``None``. Default:
                ``None``

        Returns:
            Tensor: ``data`` transformed to the image domain.
        """
        if smaps is not None:
            if not smaps.dtype == data.dtype:
                raise TypeError("data dtype does not match smaps dtype.")

        is_complex = True
        if not data.is_complex():
            if not data.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError("For real inputs, last dimension must be size 2.")

                smaps = torch.view_as_complex(smaps)

            is_complex = False
            data = torch.view_as_complex(data)

        if interp_mats is not None:
            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)

            output = tkbnF.kb_spmat_nufft_adjoint(
                data=data,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                interp_mats=interp_mats,
                norm=norm,
            )
        else:
            tables = []
            for i in range(len(self.im_size)):  # type: ignore
                tables.append(getattr(self, f"table_{i}"))

            assert isinstance(self.scaling_coef, Tensor)
            assert isinstance(self.im_size, Tensor)
            assert isinstance(self.grid_size, Tensor)
            assert isinstance(self.n_shift, Tensor)
            assert isinstance(self.numpoints, Tensor)
            assert isinstance(self.table_oversamp, Tensor)
            assert isinstance(self.offsets, Tensor)

            output = tkbnF.kb_table_nufft_adjoint(
                data=data,
                scaling_coef=self.scaling_coef,
                im_size=self.im_size,
                grid_size=self.grid_size,
                omega=omega,
                tables=tables,
                n_shift=self.n_shift,
                numpoints=self.numpoints,
                table_oversamp=self.table_oversamp,
                offsets=self.offsets.to(torch.long),
                norm=norm,
            )

        if smaps is not None:
            output = torch.sum(output * smaps.conj(), dim=1, keepdim=True)

        if not is_complex:
            output = torch.view_as_real(output)

        return output


class ToepNufft(torch.nn.Module):
    r"""Forward/backward NUFFT with Toeplitz embedding.

    This module applies :math:`Tx`, where :math:`T` is a matrix such that
    :math:`T \approx A'A`, where :math:`A` is a NUFFT matrix. Using Toeplitz
    embedding, this module approximates the :math:`A'A` operation without
    interpolations, which is extremely fast.

    The module is intended to be used in combination with an fft kernel
    computed to be the frequency response of an embedded Toeplitz matrix. The
    kernel is calculated offline via

    ``torchkbnufft.calculate_toeplitz_kernel``

    The corresponding kernel is then passed to this module in its forward
    forward operation, which applies a (zero-padded) fft filter using the
    kernel.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        image: Tensor,
        kernel: Tensor,
        smaps: Optional[Tensor] = None,
        norm: Optional[str] = "ortho",
    ) -> Tensor:
        """Toeplitz NUFFT forward function.

        Args:
            image (Tensor): The image to apply the forward/backward
                Toeplitz-embedded NUFFT to.
            kernel (Tensor): The filter response taking into account Toeplitz
                embedding.
            norm (string, optional): Whether to apply normalization with the
                FFT operation. Options are ``"ortho"`` or ``None``. Default:
                ``"ortho"``

        Returns:
            Tensor: ``image`` after applying the Toeplitz NUFFT.
        """
        if smaps is not None:
            if not smaps.dtype == image.dtype:
                raise TypeError("image dtype does not match smaps dtype.")

        is_complex = True
        if not image.is_complex():
            if not image.shape[-1] == 2:
                raise ValueError("For real inputs, last dimension must be size 2.")
            if smaps is not None:
                if not smaps.shape[-1] == 2:
                    raise ValueError("For real inputs, last dimension must be size 2.")

                smaps = torch.view_as_complex(smaps)

            is_complex = False
            image = torch.view_as_complex(image)

        if smaps is None:
            output = tkbnF.fft_filter(image=image, kernel=kernel, norm=norm)
        else:
            output = []

            # do a batch loop to conserve memory
            for (mini_image, smap) in zip(image, smaps):
                mini_image = mini_image.unsqueeze(0) * smap.unsqueeze(0)
                mini_image = tkbnF.fft_filter(
                    image=mini_image, kernel=kernel, norm=norm
                )
                mini_image = torch.sum(
                    mini_image * smap.unsqueeze(0).conj(),
                    dim=1,
                    keepdim=True,
                )
                output.append(mini_image.squeeze(0))

            output = torch.stack(output)

        if not is_complex:
            output = torch.view_as_real(output)

        return output
