"""Package info"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fastmri")
except PackageNotFoundError:
    # package is not installed
    import warnings

    warnings.warn("Could not retrieve fastmri version!")

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
