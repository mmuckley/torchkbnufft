.. torchkbnufft documentation master file, created by
   sphinx-quickstart on Tue Nov 19 12:46:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchKbNufft Documentation
========================================

`Documentation <https://torchkbnufft.readthedocs.io>`_ | `GitHub <https://github.com/mmuckley/torchkbnufft>`_ | `Notebook Examples <https://github.com/mmuckley/torchkbnufft/tree/master/notebooks>`_

About
-----

:py:mod:`torchkbnufft` implements a non-uniform Fast Fourier Transform 
[`1 <https://doi.org/10.1109/TSP.2002.807005>`_,
`2 <https://doi.org/10.1109/TMI.2005.848376>`_] with
Kaiser-Bessel gridding in PyTorch. The implementation is completely in Python,
facilitating flexible deployment in readable code with no compilation. NUFFT
functions are each wrapped as a :py:class:`torch.autograd.Function`, allowing
backpropagation through NUFFT operators for training neural networks.

This package was inspired in large part by the NUFFT implementation in the
`Michigan Image Reconstruction Toolbox (Matlab)
<https://github.com/JeffFessler/mirt>`_.

Installation
------------

Simple installation can be done via PyPI:

.. code-block:: bash

   pip install torchkbnufft

:py:mod:`torchkbnufft` only requires :py:mod:`numpy`, :py:mod:`scipy`, and
:py:mod:`torch` as dependencies.

Operation Modes and Stages
--------------------------

The package has three major classes of NUFFT operation mode: table-based NUFFT
interpolation, sparse matrix-based NUFFT interpolation, and forward/backward
operators with Toeplitz-embedded FFTs
[`3 <https://doi.org/10.1007/s002110050101>`_]. Table interpolation is the
standard operation mode, whereas the Toeplitz method is always the
fastest for forward/backward NUFFTs. For some problems, sparse matrices may be
fast. It is generally best to start with Table interpolation and then experiment
with the other modes for your problem.

Sensitivity maps can be incorporated by passing them into a
:py:class:`~torchkbnufft.KbNufft` or :py:class:`~torchkbnufft.KbNufftAdjoint`
object. Auxiliary functions for calculating sparse interpolation matrices,
density compensation functions, and Toeplitz filter kernels are also included.

For examples, see :doc:`basic`.

References
-----------

1. Fessler, J. A., & Sutton, B. P. (2003). `Nonuniform fast Fourier transforms using min-max interpolation <https://doi.org/10.1109/TSP.2002.807005>`_. *IEEE Transactions on Signal Processing*, 51(2), 560-574.

2. Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005). `Rapid gridding reconstruction with a minimal oversampling ratio <https://doi.org/10.1109/TMI.2005.848376>`_. *IEEE Transactions on Medical Imaging*, 24(6), 799-808.

3. Feichtinger, H. G., Gr, K., & Strohmer, T. (1995). `Efficient numerical methods in non-uniform sampling theory <https://doi.org/10.1007/s002110050101>`_. *Numerische Mathematik*, 69(4), 423-440.

.. toctree::
   :hidden:
   :caption: User Guide

   basic
   performance

.. toctree::
   :hidden:
   :caption: API

   torchkbnufft
   torchkbnufft.functional

.. toctree::
   :hidden:
   :caption: Core Modules

   generated/torchkbnufft.KbInterp
   generated/torchkbnufft.KbInterpAdjoint
   generated/torchkbnufft.KbNufft
   generated/torchkbnufft.KbNufftAdjoint
   generated/torchkbnufft.ToepNufft

.. toctree::
   :hidden:
   :caption: Utility Functions

   generated/torchkbnufft.calc_density_compensation_function
   generated/torchkbnufft.calc_tensor_spmatrix
   generated/torchkbnufft.calc_toeplitz_kernel

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
