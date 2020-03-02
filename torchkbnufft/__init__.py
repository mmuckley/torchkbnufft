"""Package info"""

__version__ = '0.3.2'
__author__ = 'Matthew Muckley'
__author_email__ = 'Matthew.Muckley@nyulangone.org'
__license__ = 'MIT'
__homepage__ = 'https://github.com/mmuckley/torchkbnufft'
__docs__ = 'A robust, easy-to-deploy non-uniform Fast Fourier Transform in PyTorch.'

try:
    # This variable is injected in the __builtins__ by the build
    # process.
    __TORCHKBNUFFT_SETUP__
except NameError:
    __TORCHKBNUFFT_SETUP__ = False

if __TORCHKBNUFFT_SETUP__:
    import sys
    sys.stderr.write('Partial import of during the build process.\n')
else:
    from .kbinterp import KbInterpBack, KbInterpForw
    from .kbnufft import KbNufft, AdjKbNufft, ToepNufft
    from .mrisensenufft import MriSenseNufft, AdjMriSenseNufft, ToepSenseNufft
    from .nufft import utils as nufft_utils

    __all__ = [
        'KbInterpForw',
        'KbInterpBack',
        'KbNufft',
        'AdjKbNufft',
        'MriSenseNufft',
        'AdjMriSenseNufft'
    ]
