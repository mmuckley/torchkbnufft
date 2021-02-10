Performance Tips
================

:py:mod:`torchkbnufft` is primarily written for the goal of scaling parallelism within
the PyTorch framework. There are a few dimensions on which parallelsim scales well,
such as:

1. Scaling the batch dimension.
2. Scaling the coil dimension.

Generally, you can just add to these dimensions and the package will perform well. Here
we also mention some other strategies.

Using Batched K-space Trajectories
----------------------------------

As of version ``1.1.0``, :py:mod:`torchkbnufft` can use batched k-space trajectories.
If you pass in a variable for ``omega`` with dimensions
``(N, length(im_size), klength)``, the package will parallelize the execution of all
trajectories in the ``N`` dimension. This is very useful when ``N`` is very large, as
might occur in dynamic imaging settings. The following shows an example:

.. code-block:: python

    import torch
    import torchkbnufft as tkbn
    import numpy as np
    from skimage.data import shepp_logan_phantom

    batch_size = 12

    x = shepp_logan_phantom().astype(np.complex)
    im_size = x.shape
    # convert to tensor, unsqueeze batch and coil dimension
    # output size: (batch_size, 1, ny, nx)
    x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(torch.complex64)
    x = x.repeat(batch_size, 1, 1, 1)

    klength = 64
    ktraj = np.stack(
        (np.zeros(64), np.linspace(-np.pi, np.pi, klength))
    )
    # convert to tensor, unsqueeze batch dimension
    # output size: (batch_size, 2, klength)
    ktraj = torch.tensor(ktraj).to(torch.float)
    ktraj = ktraj.unsqueeze(0).repeat(batch_size, 1, 1)

    nufft_ob = tkbn.KbNufft(im_size=im_size)
    # outputs a (batch_size, 1, klength) vector of k-space data
    kdata = nufft_ob(x, ktraj)

This code will then compute the 12 different radial spokes while parallelizing as much
as possible.

Lowering the Precision
----------------------

A very simple way to save both memory and compute time is to decrease the precision.
PyTorch normally operates at a default 32-bit floating point precision, but if you're
converting data from NumPy then you might have some data at 64-bit floating precision.
In many cases, the tradeoff for going from 64-bit to 32-bit is not severe, so you can
securely use 32-bit precision.

Lowering the Oversampling Ratio
-------------------------------

If you create a :py:class:`~torchkbnufft.KbNufft` object using the following code:

.. code-block:: python

    forw_ob = tkbn.KbNufft(im_size=im_size)

then by default it will use a 2-factor oversampled grid. For some applications, this can
be overkill. If you can sacrifice some accuracy for your application, you can use a
smaller grid with 1.25-factor oversampling:

.. code-block:: python

    grid_size = tuple([int(el * 1.25) for el in im_size])
    forw_ob = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)

Using Fewer Interpolation Neighbors
-----------------------------------

Another major speed factor is how many neighbors you use for interpolation. By default,
:py:mod:`torchkbnufft` uses 6 nearest neighbors in each dimension. If you can sacrifice
accuracy, you can get more speed by using fewer neighbors:

.. code-block:: python

    forw_ob = tkbn.KbNufft(im_size=im_size, numpoints=4)

If you know that you can be less accurate in one dimension (e.g., the z-dimension), then
you can use less neighbors in only that dimension:

.. code-block:: python

    forw_ob = tkbn.KbNufft(im_size=im_size, numpoints=(4, 6, 6))

Poor Scaling Domains
--------------------

As a high-level NUFFT implementation, we are constrained by PyTorch on areas where we
scale well. As mentioned earlier, batches and coils scale pretty well. Where we don't
scale well are:

1. Longer k-space trajectories.
2. More imaging dimensions (e.g., 3D)

For these settings, you can try to use some of the strategies here (lowering precision,
fewer neighbors, smaller grid). If you're still waiting too long for compute after
trying all of these, you may be running into the limits of the package.
