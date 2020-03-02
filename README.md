# Torch KB-NUFFT

[API](https://torchkbnufft.readthedocs.io) | [GitHub](https://github.com/mmuckley/torchkbnufft) | [Notebook Examples](https://github.com/mmuckley/torchkbnufft/tree/master/notebooks) | [Sedona Workshop Demo](https://github.com/mmuckley/torchkbnufft_demo)

Simple installation from PyPI:

```bash
pip install torchkbnufft
```

## About

Torch KB-NUFFT implements a non-uniform Fast Fourier Transform [1, 2] with Kaiser-Bessel gridding in PyTorch. The implementation is completely in Python, facilitating robustness and flexible deployment in human-readable code. NUFFT functions are each wrapped as a ```torch.autograd.Function```, allowing backpropagation through NUFFT operators for training neural networks.

This package was inspired in large part by the implementation of NUFFT operations in the Matlab version of the Michigan Image Reconstruction Toolbox, available at <https://web.eecs.umich.edu/~fessler/code/index.html>.

### Operation Modes and Stages

The package has three major classes of NUFFT operation mode: table-based NUFFT interpolation, sparse matrix-based NUFFT interpolation, and forward/backward operators with Toeplitz-embedded FFTs [3]. In most cases, computation speed follows

table < sparse matrix < Toeplitz embedding,

but better computation speed can require increased memory usage.

In addition to the three main operation modes, the package separates SENSE-NUFFT operations into three stages that can be used individually as PyTorch modules: interpolation (```torchkbnufft.KbInterp```), NUFFT (```torchkbnufft.KbNufft```), and SENSE-NUFFT (```torchkbnufft.MriSenseNufft```). The interpolation modules only apply interpolation (without scaling coefficients). The NUFFT applies the full the NUFFT for a single image into non-Cartesian k-space, including scaling coefficients. The SENSE-NUFFT can be used to include sensitivity coil multiplications, which by default at a lower level will use PyTorch broadcasting backends that enable faster multiplications across the coils.

Where appropriate, each of the NUFFT stages can be used with each NUFFT operation mode. Simple examples follow.

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
import numpy as np
from torchkbnufft import KbNufft
from skimage.data import shepp_logan_phantom

x = shepp_logan_phantom().astype(np.complex)
im_size = x.shape
x = np.stack((np.real(x), np.imag(x)))
# convert to tensor, unsqueeze batch and coil dimension
# output size: (1, 1, 2, ny, nx)
x = torch.tensor(x).unsqueeze(0).unsqueeze(0)

klength = 64
ktraj = np.stack(
    (np.zeros(64), np.linspace(-np.pi, np.pi, klength))
)
# convert to tensor, unsqueeze batch dimension
# output size: (1, 2, klength)
ktraj = torch.tensor(ktraj).unsqueeze(0)

nufft_ob = KbNufft(im_size=im_size)
# outputs a (1, 1, 2, klength) vector of k-space data
kdata = nufft_ob(x, ktraj)
```

A detailed example of basic NUFFT usage is included in ```notebooks/Basic Example.ipynb```.

### SENSE-NUFFT

The package also includes utilities for working with SENSE-NUFFT operators. The above code can be modified to replace the ```nufft_ob``` with the following ```sensenufft_ob```:

```python
from torchkbnufft import MriSenseNufft

smap = torch.rand(1, 8, 2, 400, 400)
sensenufft_ob = MriSenseNufft(im_size=im_size, smap=smap)
sense_data = sensenufft_ob(x, ktraj)
```

Application of the object in place of ```nufft_ob``` above would first multiply by the sensitivity coils in ```smap```, then compute a 64-length radial spoke for each coil. All operations are broadcast across coils, which minimizes interaction with the Python interpreter, helping computation speed.

A detailed example of SENSE-NUFFT usage is included in ```notebooks/SENSE Example.ipynb```.

### Sparse Matrix Precomputation

Sparse matrices are a fast operation mode on the CPU and for large problems at the cost of more memory usage. The following code calculates sparse interpolation matrices and uses them to compute a single radial spoke of k-space data:

```python
from torchkbnufft import AdjKbNufft
from torchkbnufft.nufft.sparse_interp_mat import precomp_sparse_mats

adjnufft_ob = AdjKbNufft(im_size=im_size)

# precompute the sparse interpolation matrices
real_mat, imag_mat = precomp_sparse_mats(ktraj, adjnufft_ob)
interp_mats = {
    'real_interp_mats': real_mat,
    'imag_interp_mats': imag_mat
}

# use sparse matrices in adjoint
image = adjnufft_ob(kdata, ktraj, interp_mats)
```

A detailed example of sparse matrix precomputation usage is included in ```notebooks/Sparse Matrix Example.ipynb```.

### Toeplitz Embedding

The package includes routines for calculating embedded Toeplitz kernels and using them as FFT filters for the forward/backward NUFFT operations [3]. This is very useful for gradient descent algorithms that must use the forward/backward ops in calculating the gradient. The following minimalist code shows an example:

```python
from torchkbnufft import AdjKbNufft, ToepNufft
from torchkbnufft.nufft.toep_functions import calc_toep_kernel

adjnufft_ob = AdjKbNufft(im_size=im_size)
toep_ob = ToepNufft()

# precompute the embedded Toeplitz FFT kernel
kern = calc_toep_kernel(adjnufft_ob, ktraj)

# use FFT kernel from embedded Toeplitz matrix
image = toep_ob(image, kern)
```

A detailed example of sparse matrix precomputation usage is included in ```notebooks/Toeplitz Example.ipynb```.

### Running on the GPU

All of the examples included in this repository can be run on the GPU by sending the NUFFT object and data to the GPU prior to the function call, e.g.:

```python
adjnufft_ob = adjnufft_ob.to(torch.device('cuda'))
kdata = kdata.to(torch.device('cuda'))
ktraj = ktraj.to(torch.device('cuda'))

image = adjnufft_ob(kdata, ktraj)
```

Similar to programming low-level code, PyTorch will throw errors if the underlying ```dtype``` and ```device``` of all objects are not matching. Be sure to make sure your data and NUFFT objects are on the right device and in the right format to avoid these errors.

## Computation Speed

TorchKbNufft is first and foremost designed to be lightweight with minimal dependencies outside of PyTorch. Speed compared to other packages depends on problem size and usage mode - generally, favorable performance can be observed with large problems (2-3 times faster than some packages with 64 coils) when using spare matrices, whereas unfavorable performance occurs with small problems in table interpolation mode (2-3 times as slow as other packages).

The following computation times in seconds were observed on a workstation with a Xeon E5-1620 CPU and an Nvidia GTX 1080 GPU for a 15-coil, 405-spoke 2D radial problem. CPU computations were done with 64-bit floats, whereas GPU computations were done with 32-bit floats (v0.2.2).

(n) = normal, (spm) = sparse matrix, (toep) = Toeplitz embedding, (f/b) = forward/backward combined

| Operation      | CPU (n) | CPU (spm) | CPU (toep)  | GPU (n)  | GPU (spm) | GPU (toep)     |
| -------------- | -------:| ---------:| -----------:| --------:| ---------:| --------------:|
| Forward NUFFT  | 3.57    | 1.61      | 0.145 (f/b) | 1.00e-01 | 1.57e-01  | 5.16e-03 (f/b) |
| Adjoint NUFFT  | 4.52    | 1.04      | N/A         | 1.06e-01 | 1.63e-01  | N/A            |

Profiling for your machine can be done by running

```python
python profile_torchkbnufft.py
```

## Other Packages

For users interested in NUFFT implementations for other computing platforms, the following is a partial list of other projects:

1. [TF KB-NUFFT](https://github.com/zaccharieramzi/tfkbnufft) (KB-NUFFT for TensorFlow)
2. [SigPy](https://github.com/mikgroup/sigpy) (for Numpy arrays, Numba (for CPU) and CuPy (for GPU) backends)
3. [FINUFFT](https://github.com/flatironinstitute/finufft) (for MATLAB, Python, Julia, C, etc., very efficient)
4. [NFFT](https://github.com/NFFT/nfft) (for Julia)

## References

1. Fessler, J. A., & Sutton, B. P. (2003). Nonuniform fast Fourier transforms using min-max interpolation. *IEEE transactions on signal processing*, 51(2), 560-574.

2. Beatty, P. J., Nishimura, D. G., & Pauly, J. M. (2005). Rapid gridding reconstruction with a minimal oversampling ratio. *IEEE transactions on medical imaging*, 24(6), 799-808.

3. Feichtinger, H. G., Gr, K., & Strohmer, T. (1995). Efficient numerical methods in non-uniform sampling theory. Numerische Mathematik, 69(4), 423-440.

## Citation

If you want to cite the package, you can use any of the following:

```bibtex
@conference{muckley:20:tah,
  author = {M. J. Muckley and R. Stern and T. Murrell and F. Knoll},
  title = {{TorchKbNufft}: A High-Level, Hardware-Agnostic Non-Uniform Fast Fourier Transform},
  booktitle = {ISMRM Workshop on Data Sampling \& Image Reconstruction},
  year = 2020
}

@misc{Muckley2019,
  author = {Muckley, M.J. et al.},
  title = {Torch KB-NUFFT},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mmuckley/torchkbnufft}}
}
```
