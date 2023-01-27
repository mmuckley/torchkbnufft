# torchkbnufft

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![CI Badge](https://github.com/mmuckley/torchkbnufft/workflows/Build%20and%20test/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/torchkbnufft/badge/?version=stable)](https://torchkbnufft.readthedocs.io/en/stable/?badge=latest) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Basic%20Example.ipynb)

[Documentation](https://torchkbnufft.readthedocs.io) | [GitHub](https://github.com/mmuckley/torchkbnufft) | [Notebook Examples](#examples)

Simple installation from PyPI:

```bash
pip install torchkbnufft
```

## About

`torchkbnufft` implements a non-uniform Fast Fourier Transform
[[1, 2](#references)] with Kaiser-Bessel gridding in PyTorch. The
implementation is completely in Python, facilitating flexible deployment in
readable code with no compilation. NUFFT functions are each wrapped as a
```torch.autograd.Function```, allowing backpropagation through NUFFT operators
for training neural networks.

This package was inspired in large part by the NUFFT implementation in the
[Michigan Image Reconstruction Toolbox (Matlab)](https://github.com/JeffFessler/mirt).

### Operation Modes and Stages

The package has three major classes of NUFFT operation mode: table-based NUFFT
interpolation, sparse matrix-based NUFFT interpolation, and forward/backward
operators with Toeplitz-embedded FFTs [[3](#references)]. Roughly, computation
speed follows:

| Type          | Speed                  |
| ------------- | ---------------------- |
| Toeplitz      | Fastest                |
| Table         | Medium                 |
| Sparse Matrix | Slow (not recommended) |

It is generally best to start with Table interpolation and then experiment with
the other modes for your problem.

Sensitivity maps can be incorporated by passing them into a `KbNufft` or
`KbNufftAdjoint` object.

## Documentation

An html-based documentation reference on
[Read the Docs](https://torchkbnufft.readthedocs.io).

Most files are accompanied with docstrings that can be read with ```help```
while running IPython. Example:

```python
from torchkbnufft import KbNufft

help(KbNufft)
```

## Examples

`torchkbnufft` can be used for N-D NUFFT transformations. The examples here
start with a simple 2D NUFFT, then expand it to SENSE (a task with multiple,
parallel 2D NUFFTs).

The last two examples demonstrate NUFFTs based on sparse matrix multiplications
(which can be useful for high-dimensional cases) and Toeplitz NUFFTs (which are
an extremely fast forward-backward NUFFT technique).

All examples have associated notebooks that you can run in Google Colab:

- [Basic Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Basic%20Example.ipynb)
- [SENSE-NUFFT Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/SENSE%20Example.ipynb)
- [Sparse Matrix Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Sparse%20Matrix%20Example.ipynb)
- [Toeplitz Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Toeplitz%20Example.ipynb)

### Simple Forward NUFFT

[Basic NUFFT Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Basic%20Example.ipynb)

The following code loads a Shepp-Logan phantom and computes a single radial
spoke of k-space data:

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
# output size: (2, klength)
ktraj = torch.tensor(ktraj, dtype=torch.float)

nufft_ob = tkbn.KbNufft(im_size=im_size)
# outputs a (1, 1, klength) vector of k-space data
kdata = nufft_ob(x, ktraj)
```

### SENSE-NUFFT

[SENSE-NUFFT Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/SENSE%20Example.ipynb)

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

### Sparse Matrix Precomputation

[Sparse Matrix Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Sparse%20Matrix%20Example.ipynb)

Sparse matrices are an alternative to table interpolation. Their speed can
vary, but they are a bit more accurate than standard table mode. The following
code calculates sparse interpolation matrices and uses them to compute a single
radial spoke of k-space data:

```python
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size)

# precompute the sparse interpolation matrices
interp_mats = tkbn.calc_tensor_spmatrix(
    ktraj,
    im_size=im_size
)

# use sparse matrices in adjoint
image = adjnufft_ob(kdata, ktraj, interp_mats)
```

Sparse matrix multiplication is only implemented for real numbers in PyTorch,
which can limit their speed.

### Toeplitz Embedding

[Toeplitz Example in Colab](https://colab.research.google.com/github/mmuckley/torchkbnufft/blob/main/notebooks/Toeplitz%20Example.ipynb)

The package includes routines for calculating embedded Toeplitz kernels and
using them as FFT filters for the forward/backward NUFFT operations
[[3](#references)]. This is very useful for gradient descent algorithms that
must use the forward/backward ops in calculating the gradient. The following
code shows an example:

```python
toep_ob = tkbn.ToepNufft()

# precompute the embedded Toeplitz FFT kernel
kernel = tkbn.calc_toeplitz_kernel(ktraj, im_size)

# use FFT kernel from embedded Toeplitz matrix
image = toep_ob(image, kernel)
```

### Running on the GPU

All of the examples included in this repository can be run on the GPU by
sending the NUFFT object and data to the GPU prior to the function call, e.g.,

```python
adjnufft_ob = adjnufft_ob.to(torch.device('cuda'))
kdata = kdata.to(torch.device('cuda'))
ktraj = ktraj.to(torch.device('cuda'))

image = adjnufft_ob(kdata, ktraj)
```

PyTorch will throw errors if the underlying ```dtype``` and ```device``` of all
objects are not matching. Be sure to make sure your data and NUFFT objects are
on the right device and in the right format to avoid these errors.

## Computation Speed

The following computation times in seconds were observed on a workstation with
a Xeon E5-2698 CPU and an Nvidia Quadro GP100 GPU for a 15-coil, 405-spoke 2D
radial problem. CPU computations were limited to 8 threads and done with 64-bit
floats, whereas GPU computations were done with 32-bit floats. The benchmark
used `torchkbnufft` version 1.0.0 and `torch` version 1.7.1.

(n) = normal, (spm) = sparse matrix, (toep) = Toeplitz embedding, (f/b) = forward/backward combined

| Operation      | CPU (n) | CPU (spm) | CPU (toep)  | GPU (n)  | GPU (spm) | GPU (toep)     |
| -------------- | -------:| ---------:| -----------:| --------:| ---------:| --------------:|
| Forward NUFFT  | 0.82    | 0.77      | 0.058 (f/b) | 2.58e-02 | 7.44e-02  | 3.03e-03 (f/b) |
| Adjoint NUFFT  | 0.75    | 0.76      | N/A         | 3.56e-02 | 7.93e-02  | N/A            |

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

1. Fessler, J. A., & Sutton, B. P. (2003). [Nonuniform fast Fourier transforms using min-max interpolation](https://doi.org/10.1109/TSP.2002.807005). *IEEE Transactions on Signal Processing*, 51(2), 560-574.

2. Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005). [Rapid gridding reconstruction with a minimal oversampling ratio](https://doi.org/10.1109/TMI.2005.848376). *IEEE Transactions on Medical Imaging*, 24(6), 799-808.

3. Feichtinger, H. G., Gr, K., & Strohmer, T. (1995). [Efficient numerical methods in non-uniform sampling theory](https://doi.org/10.1007/s002110050101). *Numerische Mathematik*, 69(4), 423-440.

## Citation

If you use the package, please cite:

```bibtex
@conference{muckley:20:tah,
  author = {M. J. Muckley and R. Stern and T. Murrell and F. Knoll},
  title = {{TorchKbNufft}: A High-Level, Hardware-Agnostic Non-Uniform Fast {Fourier} Transform},
  booktitle = {ISMRM Workshop on Data Sampling \& Image Reconstruction},
  year = 2020,
  note = {Source code available at https://github.com/mmuckley/torchkbnufft},
}
```
