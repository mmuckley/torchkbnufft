import numpy as np


def mrisensesim(size, ncoils=8, array_cent=None, coil_width=2, n_rings=None, phi=0):
    """Apply simulated sensitivity maps. Based on a script by Florian Knoll.

    Args:
        size (tuple): Size of the image array for the sensitivity coils.
        nc_range (int, default: 8): Number of coils to simulate.
        array_cent (tuple, default: 0): Location of the center of the coil
            array.
        coil_width (double, default: 2): Parameter governing the width of the
            coil, multiplied by actual image dimension.
        n_rings (int, default: ncoils // 4): Number of rings for a
            cylindrical hardware set-up.
        phi (double, default: 0): Parameter for rotating coil geometry.

    Returns:
        coil_array: An array of dimensions (ncoils (N)), specifying
            spatially-varying sensitivity maps for each coil.
    """
    if array_cent is None:
        c_shift = [0, 0, 0]
    elif len(array_cent) < 3:
        c_shift = array_cent + (0,)
    else:
        c_shift = array_cent

    c_width = coil_width * min(size)

    if (len(size) > 2):
        if n_rings is None:
            n_rings = ncoils // 4

    c_rad = min(size[0:1]) / 2
    smap = []
    if (len(size) > 2):
        zz, yy, xx = np.meshgrid(range(size[2]), range(size[1]),
                                 range(size[0]), indexing='ij')
    else:
        yy, xx = np.meshgrid(range(size[1]), range(size[0]),
                             indexing='ij')

    if ncoils > 1:
        x0 = np.zeros((ncoils,))
        y0 = np.zeros((ncoils,))
        z0 = np.zeros((ncoils,))

        for i in range(ncoils):
            if (len(size) > 2):
                theta = np.radians((i - 1) * 360 / (ncoils + n_rings) + phi)
            else:
                theta = np.radians((i - 1) * 360 / ncoils + phi)
            x0[i] = c_rad * np.cos(theta) + size[0] / 2
            y0[i] = c_rad * np.sin(theta) + size[1] / 2
            if (len(size) > 2):
                z0[i] = (size[2] / (n_rings + 1)) * (i // n_rings)
                smap.append(np.exp(-1 * ((xx - x0[i])**2 + (yy - y0[i])**2 +
                                         (zz - z0[i])**2) / (2 * c_width)))
            else:
                smap.append(np.exp(-1 * ((xx - x0[i])**2 + (yy - y0[i])**2) /
                                   (2 * c_width)))
    else:
        x0 = c_shift[0]
        y0 = c_shift[1]
        z0 = c_shift[2]
        if (len(size) > 2):
            smap = np.exp(-1 * ((xx - x0)**2 + (yy - y0)**2 +
                                (zz - z0)**2) / (2 * c_width))
        else:
            smap = np.exp(-1 * ((xx - x0)**2 + (yy - y0)**2) / (2 * c_width))

    side_mat = np.arange(int(size[0] // 2) - 20, 1, -1)
    side_mat = (np.reshape(side_mat, (1,) + side_mat.shape) *
                np.ones(shape=(size[1], 1)))
    cent_zeros = np.zeros(shape=(size[1], size[0] - side_mat.shape[1] * 2))

    ph = np.concatenate((side_mat, cent_zeros, side_mat), axis=1) / 10
    if (len(size) > 2):
        ph = np.reshape(ph, (1,) + ph.shape)

    for i, s in enumerate(smap):
        smap[i] = s * np.exp(i * 1j * ph * np.pi / 180)

    return smap
