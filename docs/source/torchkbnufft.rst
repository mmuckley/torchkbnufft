torchkbnufft
============

.. currentmodule:: torchkbnufft

NUFFT Modules
----------------------------------

These are the primary workhorse modules for applying NUFFT operations.

.. autosummary::
    :toctree: generated
    :nosignatures:

    KbInterp
    KbInterpAdjoint
    KbNufft
    KbNufftAdjoint
    ToepNufft

Auxiliary Functions
----------------------------------

Functions for calcualting density compensation and Toeplitz kernels.

.. autosummary::
    :toctree: generated
    :nosignatures:

    calculate_density_compensation_function
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


KbInterp
~~~~~~~~

.. autoclass:: KbInterp
    :members:

KbInterpAdjoint
~~~~~~~~~~~~~~~

.. autoclass:: KbInterpAdjoint
    :members:

KbNufft
~~~~~~~

.. autoclass:: KbNufft
    :members:

KbNufftAdjoint
~~~~~~~~~~~~~~

.. autoclass:: KbNufftAdjoint
    :members:

ToepNufft
~~~~~~~~~

.. autoclass:: ToepNufft
    :members:

calculate_density_compensation_function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_density_compensation_function

calculate_toeplitz_kernel
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_toeplitz_kernel

absolute
~~~~~~~~

.. autofunction:: absolute

complex_mult
~~~~~~~~~~~~

.. autofunction:: complex_mult

complex_sign
~~~~~~~~~~~~

.. autofunction:: complex_sign

conj_complex_mult
~~~~~~~~~~~~~~~~~

.. autofunction:: conj_complex_mult

imag_exp
~~~~~~~~

.. autofunction:: imag_exp

inner_product
~~~~~~~~~~~~~

.. autofunction:: inner_product
