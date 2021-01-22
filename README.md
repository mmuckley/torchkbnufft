# Torch KB-NUFFT

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md)
![CI Badge](https://github.com/mmuckley/torchkbnufft/workflows/Build%20and%20test/badge.svg?branch=master)

[API](https://torchkbnufft.readthedocs.io) | [GitHub](https://github.com/mmuckley/torchkbnufft) | [Notebook Examples](https://github.com/mmuckley/torchkbnufft/tree/master/notebooks) | [Sedona Workshop Demo](https://github.com/mmuckley/torchkbnufft_demo)

Simple installation from PyPI:

```bash
pip install torchkbnufft
```

## About

Torch KB-NUFFT implements a non-uniform Fast Fourier Transform [1, 2] with
Kaiser-Bessel gridding in PyTorch. The implementation is completely in Python,
facilitating robustness and flexible deployment in human-readable code. NUFFT
functions are each wrapped as a ```torch.autograd.Function```, allowing
backpropagation through NUFFT operators for training neural networks.

This package was inspired in large part by the implementation of NUFFT
operations in the
[Michigan Image Reconstruction Toolbox (Matlab)](https://github.com/JeffFessler/mirt).

### Operation Modes and Stages

The package has three major classes of NUFFT operation mode: table-based NUFFT
interpolation, sparse matrix-based NUFFT interpolation, and forward/backward
operators with Toeplitz-embedded FFTs [3]. Roughly, computation speed follows:

| Type          | Speed                                                      |
------------------------------------------------------------------------------
| Toeplitz      | Fastest                                                    |
| Sparse Matrix | Slow (CPU, small coil count), Fast (GPU, large coil count) |
| Table         | Fast (CPU, small coil count), Slow (GPU, large coil count) |

It is generally best to start with Table interpolation and then experiment with
the other modes for your problem.

Sensitivity maps can be incorporated by passing them into a `KbNufft` or
`KbNufftAdjoint` object

## Documentation

Most files are accompanied with docstrings that can be read with ```help``` while running IPython. Example:

```python
from torchkbnufft import KbNufft

help(KbNufft)
```

Behavior can also be inferred by inspecting the source code [here](https://github.com/mmuckley/torchkbnufft). An html-based API reference is [here](https://torchkbnufft.readthedocs.io).

## Examples

### Simple Forward NUFFT

Jupyter notebook examples are in the ```notebooks/``` folder. The following minimalist code loads a Shepp-Logan phantom and computes a single radial spoke of k-space data:

```python
import torch
import torchkbnufft as tkbn
import numpy as np
from skimage.data import shepp_logan_phantom

x = shepp_logan_phantom().astype(np.complex)
im_size = x.shape
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, ny, nx)
x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(torch.complex64)

klength = 64
ktraj = np.stack(
    (np.zeros(64), np.linspace(-np.pi, np.pi, klength))
)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = torch.tensor(ktraj).to(torch.float)

nufft_ob = tkbn.KbNufft(im_size=im_size)
# outputs a (1, 1, 2, klength) vector of k-space data
kdata = nufft_ob(x, ktraj)
```

A detailed example of basic NUFFT usage is
[here](notebooks/Basic%20Example.ipynb).

### SENSE-NUFFT

The package also includes utilities for working with SENSE-NUFFT operators. The
above code can be modified to include sensitivity maps.

```python
smaps = torch.rand(1, 8, 400, 400) + 1j * torch.rand(1, 8, 400, 400)
sense_data = nufft_ob(x, ktraj, smaps=smaps.to(x))
```

This code first multiplies by the sensitivity coils in ```smaps```, then
computes a 64-length radial spoke for each coil. All operations are broadcast
across coils, which minimizes interaction with the Python interpreter, helping
computation speed.

A detailed example of SENSE-NUFFT usage is
[here](notebooks/SENSE%20Example.ipynb).

### Sparse Matrix Precomputation

Sparse matrices are an alternative to table interpolation. Their speed can
vary, but they are a bit more accurate than standard table mode. The following
code calculates sparse interpolation matrices and uses them to compute a single
radial spoke of k-space data:

```python
import torchkbnufft as tkbn

adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)

# precompute the sparse interpolation matrices
interp_mats = tkbn.build_tensor_spmatrix(
    ktraj,
    adjnufft_ob.numpoints.tolist(),
    adjnufft_ob.im_size.tolist(),
    adjnufft_ob.grid_size.tolist(),
    adjnufft_ob.n_shift.tolist(),
    adjnufft_ob.order.tolist(),
    adjnufft_ob.alpha.tolist(),
)
# convert to correct data type
interp_mats = tuple([t.to(torch.float) for t in interp_mats])

# use sparse matrices in adjoint
image = adjnufft_ob(kdata, ktraj, interp_mats)
```

Sparse matrix multiplication is only implemented for real numbers in PyTorch,
so you'll have to pass in floats instead of complex numbers. A detailed
example of sparse matrix precomputation usage is
[here](notebooks/Sparse%20Matrix%20Example.ipynb).

### Toeplitz Embedding

The package includes routines for calculating embedded Toeplitz kernels and
using them as FFT filters for the forward/backward NUFFT operations [3]. This
is very useful for gradient descent algorithms that must use the
forward/backward ops in calculating the gradient. The following minimalist code
shows an example:

```python
import torchkbnufft as tkbn

adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)
toep_ob = tkbn.ToepNufft()

# precompute the embedded Toeplitz FFT kernel
kernel = tkbn.calculate_toeplitz_kernel(ktraj, im_size)

# use FFT kernel from embedded Toeplitz matrix
image = toep_ob(image, kernel)
```

A detailed example of sparse matrix precomputation usage is included 
[here](notebooks/Toeplitz%20Example.ipynb).

### Running on the GPU

All of the examples included in this repository can be run on the GPU by
sending the NUFFT object and data to the GPU prior to the function call, e.g.,

```python
adjnufft_ob = adjnufft_ob.to(torch.device('cuda'))
kdata = kdata.to(torch.device('cuda'))
ktraj = ktraj.to(torch.device('cuda'))

image = adjnufft_ob(kdata, ktraj)
```

Similar to programming low-level code, PyTorch will throw errors if the
underlying ```dtype``` and ```device``` of all objects are not matching. Be
sure to make sure your data and NUFFT objects are on the right device and in
the right format to avoid these errors.

## Computation Speed

Profiling for your machine can be done by running

```python
pip install -r dev-requirements.txt
python profile_torchkbnufft.py
```

## Other Packages

For users interested in NUFFT implementations for other computing platforms,
the following is a partial list of other projects:

1. [TF KB-NUFFT](https://github.com/zaccharieramzi/tfkbnufft) (KB-NUFFT for TensorFlow)
2. [SigPy](https://github.com/mikgroup/sigpy) (for Numpy arrays, Numba (for CPU) and CuPy (for GPU) backends)
3. [FINUFFT](https://github.com/flatironinstitute/finufft) (for MATLAB, Python, Julia, C, etc., very efficient)
4. [NFFT](https://github.com/NFFT/nfft) (for Julia)
5. [PyNUFFT](https://github.com/jyhmiinlin/pynufft) (for Numpy, also has PyCUDA/PyOpenCL backends)

## References

1. Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation. *IEEE transactions on signal processing*, 51(2), 560-574.

2. Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005). Rapid gridding reconstruction with a minimal oversampling ratio. *IEEE transactions on medical imaging*, 24(6), 799-808.

3. Feichtinger, H. G., Gr, K., & Strohmer, T. (1995). Efficient numerical methods in non-uniform sampling theory. Numerische Mathematik, 69(4), 423-440.

## Citation

If you use the package, please cite:

```bibtex
@conference{muckley:20:tah,
  author = {M. J. Muckley and R. Stern and T. Murrell and F. Knoll},
  title = {{TorchKbNufft}: A High-Level, Hardware-Agnostic Non-Uniform Fast Fourier Transform},
  booktitle = {ISMRM Workshop on Data Sampling \& Image Reconstruction},
  year = 2020
}
```
