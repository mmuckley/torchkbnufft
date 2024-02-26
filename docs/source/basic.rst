Basic Usage
=============

:py:mod:`torchkbnufft` works primarily via PyTorch modules. You create a module with
the properties of your imaging setup. The module will calculate a Kaiser-Bessel
kernel and some interpolation parameters based on your inputs. Then, you apply
the module to your data stored as PyTorch tensors. NUFFT operations are wrapped in
:py:class:`torch.autograd.Function` classes for backpropagation and training
neural networks.

The following code loads a Shepp-Logan phantom and computes a single
radial spoke of k-space data:

.. code-block:: python

   import torch
   import torchkbnufft as tkbn
   import numpy as np
   from skimage.data import shepp_logan_phantom

   x = shepp_logan_phantom().astype(complex)
   im_size = x.shape
   # convert to tensor, unsqueeze batch and coil dimension
   # output size: (1, 1, ny, nx)
   x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(torch.complex64)

   klength = 64
   ktraj = np.stack(
      (np.zeros(64), np.linspace(-np.pi, np.pi, klength))
   )
   # convert to tensor, unsqueeze batch dimension
   # output size: (2, klength)
   ktraj = torch.tensor(ktraj).to(torch.float)

   nufft_ob = tkbn.KbNufft(im_size=im_size)
   # outputs a (1, 1, klength) vector of k-space data
   kdata = nufft_ob(x, ktraj)

The package also includes utilities for working with SENSE-NUFFT operators. The
above code can be modified to include sensitivity maps.

.. code-block:: python

   smaps = torch.rand(1, 8, 400, 400) + 1j * torch.rand(1, 8, 400, 400)
   sense_data = nufft_ob(x, ktraj, smaps=smaps.to(x))

This code first multiplies by the sensitivity coils in ``smaps``, then
computes a 64-length radial spoke for each coil. All operations are broadcast
across coils, which minimizes interaction with the Python interpreter, helping
computation speed.

Sparse matrices are an alternative to table interpolation. Their speed can
vary, but they are a bit more accurate than standard table mode. The following
code calculates sparse interpolation matrices and uses them to compute a single
radial spoke of k-space data:

.. code-block:: python

   adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)

   # precompute the sparse interpolation matrices
   interp_mats = tkbn.calc_tensor_spmatrix(
      ktraj,
      im_size=im_size
   )

   # use sparse matrices in adjoint
   image = adjnufft_ob(kdata, ktraj, interp_mats)

Sparse matrix multiplication is only implemented for real numbers in PyTorch,
which can limit their speed.

The package includes routines for calculating 
`embedded Toeplitz kernels <https://doi.org/10.1007/s002110050101>`_ and
using them as FFT filters for the forward/backward NUFFT operations. This is very useful
for gradient descent algorithms that must use the forward/backward ops in calculating
the gradient. The following code shows an example:

.. code-block:: python

   toep_ob = tkbn.ToepNufft()

   # precompute the embedded Toeplitz FFT kernel
   kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size)

   # use FFT kernel from embedded Toeplitz matrix
   image = toep_ob(image, kernel)

All of the examples included in this repository can be run on the GPU by
sending the NUFFT object and data to the GPU prior to the function call, e.g.,

.. code-block:: python

   adjnufft_ob = adjnufft_ob.to(torch.device('cuda'))
   kdata = kdata.to(torch.device('cuda'))
   ktraj = ktraj.to(torch.device('cuda'))

   image = adjnufft_ob(kdata, ktraj)

Similar to programming low-level code, PyTorch will throw errors if the
underlying ``dtype`` and ``device`` of all objects are not matching. Be
sure to make sure your data and NUFFT objects are on the right device and in
the right format to avoid these errors.

For more details, please examine the API in :doc:`torchkbnufft` or check out
the notebooks below on Google Colab.

- `Basic Example <https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Basic%20Example.ipynb>`_
- `SENSE-NUFFT Example <https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/SENSE%20Example.ipynb>`_
- `Sparse Matrix Example <https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Sparse%20Matrix%20Example.ipynb>`_
- `Toeplitz Example <https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Toeplitz%20Example.ipynb>`_
