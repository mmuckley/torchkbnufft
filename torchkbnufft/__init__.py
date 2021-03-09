"""Package info"""

__version__ = "1.2.0"
__author__ = "Matthew Muckley"
__author_email__ = "matt.muckley@gmail.com"
__license__ = "MIT"
__homepage__ = "https://github.com/mmuckley/torchkbnufft"
__docs__ = "A high-level, easy-to-deploy non-uniform Fast Fourier Transform in PyTorch."

try:
    # This variable is injected in the __builtins__ by the build
    # process.
    __TORCHKBNUFFT_SETUP__  # type: ignore
except NameError:
    __TORCHKBNUFFT_SETUP__ = False

if __TORCHKBNUFFT_SETUP__:
    import sys

    sys.stderr.write("Partial import of during the build process.\n")
else:
    import torchkbnufft.functional
    import torchkbnufft.modules

    from ._math import (
        absolute,
        complex_mult,
        complex_sign,
        conj_complex_mult,
        imag_exp,
        inner_product,
    )
    from ._nufft import utils as nufft_utils
    from ._nufft.dcomp import calc_density_compensation_function
    from ._nufft.spmat import calc_tensor_spmatrix
    from ._nufft.toep import calc_toeplitz_kernel
    from .modules import KbInterp, KbInterpAdjoint, KbNufft, KbNufftAdjoint, ToepNufft

    __all__ = [
        "KbInterp",
        "KbInterpAdjoint",
        "KbNufft",
        "KbNufftAdjoint",
        "ToepNufft",
        "absolute",
        "calc_density_compensation_function",
        "calc_tensor_spmatrix",
        "calc_toeplitz_kernel",
        "complex_mult",
        "complex_sign",
        "conj_complex_mult",
        "functional",
        "imag_exp",
        "inner_product",
        "modules",
    ]
