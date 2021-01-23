torchkbnufft
============

.. currentmodule:: torchkbnufft

NUFFT Modules
----------------------------------

These are the primary workhorse modules for applying NUFFT operations.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    KbInterp
    KbInterpAdjoint
    KbNufft
    KbNufftAdjoint
    ToepNufft

Utility Functions
----------------------------------

Functions for calculating density compensation and Toeplitz kernels.

.. autosummary::
    :toctree: generated
    :nosignatures:

    calculate_density_compensation_function
    calculate_tensor_spmatrix
    calculate_toeplitz_kernel

Math Functions
----------------------------------

Complex mathematical operations (gradually being removed as of PyTorch 1.7).

.. autosummary::
    :toctree: generated
    :nosignatures:

    absolute
    complex_mult
    complex_sign
    conj_complex_mult
    imag_exp
    inner_product
