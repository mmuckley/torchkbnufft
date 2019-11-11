# Torch KB-NUFFT

Simple installation from PyPI:

```bash
pip install torchkbnufft
```

## About

Torch KB-NUFFT implements a non-uniform Fast Fourier Transform with Kaiser-Bessel gridding in PyTorch. The implementation is completely in Python, facilitating robustness and flexible deployment in human-readable code. NUFFT functions are each wrapped as a ```torch.autograd.Function```, allowing backpropagation through NUFFT operators for training neural networks.

## Documentation

Most files are accompanied with docstrings that can be read with ```help``` while running IPython. Example:

```python
from torchkbnufft import KbNufft

help(KbNufft)
```

Behavior can also be inferred by inspecting the source code at ```https://github.com/mmuckley/torchkbnufft```.

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
y = nufft_ob(x, ktraj)
```

A detailed example of basic NUFFT usage is included in ```notebooks/Basic Example.ipynb```.

### SENSE-NUFFT

The package also includes utilities for working with SENSE-NUFFT operators. The above code can be modified to replace the ```nufft_ob``` with the following ```sensenufft_ob```:

```python
from torchkbnufft import MriSenseNufft

sensenufft_ob = MriSenseNufft(im_size=im_size, smap=smap)
```

Application of the object in place of ```nufft_ob``` above would first multiply by the sensitivity coils in ```smap```, then compute a 64-length radial spoke for each coil. All operations are broadcast across coils, which minimizes interaction with the Python interpreter and maximizes computation speed.

A detailed example of SENSE-NUFFT usage is included in ```notebooks/SENSE Example.ipynb```.

### Sparse Matrix Precomputation

In order to conserve memory, in normal operation mode the package includes a loop over interpolation offsets and calculates interpolation coefficients for each offset. This process can become slow due to calls to the Python interpreter. To avoid interpreter calls, one can instead precompute a sparse interpolation matrix:

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

### Running on the GPU

All of the examples included in this repository can be run on the GPU by sending the NUFFT object and data to the GPU prior to the function call, e.g.:

```python
adjnufft_ob = adjnufft_ob.to(torch.device('cuda'))
kdata = kdata.to(torch.device('cuda'))
ktraj = ktraj.to(torch.device('cuda'))

image = adjnufft_ob(kdata, ktraj)
```

Similar to programming low-level code, PyTorch will throw errors if the underlying ```dtype``` and ```device``` of all objects are not matching. Be sure to make sure your data and NUFFT objects are on the right device and in the right format to avoid these errors.

## Citation

If you want to cite the  package, you can use the following:

```bibtex
@misc{Muckley2019,
  author = {Muckley, M.J. et al.},
  title = {Torch KB-NUFFT},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mmuckley/torchkbnufft}}
}
```
