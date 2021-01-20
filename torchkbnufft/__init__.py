"""Package info"""

__version__ = "1.0.0"
__author__ = "Matthew Muckley"
__author_email__ = "matt.muckley@gmail.com"
__license__ = "MIT"
__homepage__ = "https://github.com/mmuckley/torchkbnufft"
__docs__ = "A robust, easy-to-deploy non-uniform Fast Fourier Transform in PyTorch."

try:
    # This variable is injected in the __builtins__ by the build
    # process.
    __TORCHKBNUFFT_SETUP__
except NameError:
    __TORCHKBNUFFT_SETUP__ = False

if __TORCHKBNUFFT_SETUP__:
    import sys

    sys.stderr.write("Partial import of during the build process.\n")
else:
    from .nufft.utils import build_tensor_spmatrix

    from .kbinterp import KbInterpAdjoint, KbInterpForward

    # from .kbnufft import AdjKbNufft, KbNufft, ToepNufft
    from .math import (
        absolute,
        complex_mult,
        complex_sign,
        conj_complex_mult,
        imag_exp,
        inner_product,
    )

    # from .mrisensenufft import AdjMriSenseNufft, MriSenseNufft, ToepSenseNufft
    from .nufft import utils as nufft_utils

    __all__ = [
        "KbInterpForward",
        "KbInterpAdjoint",
        # "KbNufft",
        # "AdjKbNufft",
        # "MriSenseNufft",
        # "AdjMriSenseNufft",
        "absolute",
        "complex_mult",
        "complex_sign",
        "conj_complex_mult",
        "imag_exp",
        "inner_product",
    ]
