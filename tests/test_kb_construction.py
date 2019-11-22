import torch
import numpy as np

from torchkbnufft import (AdjKbNufft, AdjMriSenseNufft, KbInterpBack,
                          KbInterpForw, KbNufft, MriSenseNufft)

norm_tol = 1e-10


def test_kb_matching():
    def check_tables(table1, table2):
        for ind, table in enumerate(table1):
            assert np.linalg.norm(table - table2[ind]) < norm_tol

    im_szs = [(256, 256), (10, 256, 256)]

    kbwidths = [2.34, 5]
    orders = [0, 2]

    for kbwidth in kbwidths:
        for order in orders:
            for im_sz in im_szs:
                smap = torch.randn(*((1,) + im_sz))

                base_table = AdjKbNufft(
                    im_sz, order=order, kbwidth=kbwidth).table

                cur_table = KbNufft(im_sz, order=order, kbwidth=kbwidth).table
                check_tables(base_table, cur_table)

                cur_table = KbInterpBack(
                    im_sz, order=order, kbwidth=kbwidth).table
                check_tables(base_table, cur_table)

                cur_table = KbInterpForw(
                    im_sz, order=order, kbwidth=kbwidth).table
                check_tables(base_table, cur_table)

                cur_table = MriSenseNufft(
                    smap, im_sz, order=order, kbwidth=kbwidth).table
                check_tables(base_table, cur_table)

                cur_table = AdjMriSenseNufft(
                    smap, im_sz, order=order, kbwidth=kbwidth).table
                check_tables(base_table, cur_table)


def test_2d_init_inputs():
    # all object initializations have assertions
    # this should result in an error if any dimensions don't match

    # test 2d scalar inputs
    im_sz = (256, 256)
    smap = torch.randn(*((1,) + im_sz))
    grid_sz = (512, 512)
    n_shift = (128, 128)
    numpoints = 6
    table_oversamp = 2**10
    kbwidth = 2.34
    order = 0
    norm = 'None'

    ob = KbInterpForw(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)
    ob = KbInterpBack(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)

    ob = KbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjKbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)

    ob = MriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjMriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)

    # test 2d tuple inputs
    im_sz = (256, 256)
    smap = torch.randn(*((1,) + im_sz))
    grid_sz = (512, 512)
    n_shift = (128, 128)
    numpoints = (6, 6)
    table_oversamp = (2**10, 2**10)
    kbwidth = (2.34, 2.34)
    order = (0, 0)
    norm = 'None'

    ob = KbInterpForw(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)
    ob = KbInterpBack(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)

    ob = KbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjKbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)

    ob = MriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjMriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)


def test_3d_init_inputs():
    # all object initializations have assertions
    # this should result in an error if any dimensions don't match

    # test 3d scalar inputs
    im_sz = (10, 256, 256)
    smap = torch.randn(*((1,) + im_sz))
    grid_sz = (10, 512, 512)
    n_shift = (5, 128, 128)
    numpoints = 6
    table_oversamp = 2**10
    kbwidth = 2.34
    order = 0
    norm = 'None'

    ob = KbInterpForw(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)
    ob = KbInterpBack(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)

    ob = KbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjKbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)

    ob = MriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjMriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)

    # test 3d tuple inputs
    im_sz = (10, 256, 256)
    smap = torch.randn(*((1,) + im_sz))
    grid_sz = (10, 512, 512)
    n_shift = (5, 128, 128)
    numpoints = (6, 6, 6)
    table_oversamp = (2**10, 2**10, 2**10)
    kbwidth = (2.34, 2.34, 2.34)
    order = (0, 0, 0)
    norm = 'None'

    ob = KbInterpForw(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)
    ob = KbInterpBack(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order)

    ob = KbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjKbNufft(
        im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)

    ob = MriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
    ob = AdjMriSenseNufft(
        smap=smap, im_size=im_sz, grid_size=grid_sz, n_shift=n_shift, numpoints=numpoints,
        table_oversamp=table_oversamp, kbwidth=kbwidth, order=order, norm=norm)
