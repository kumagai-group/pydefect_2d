# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest

from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import SlabGaussModel


@pytest.fixture
def slab_gauss_model():
    charge = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    potential = [-1.0, 1.0, 2.0, 4.0, 2.0, 1.0, -1.0, -2.0, -3.0, -2.0]
    return SlabGaussModel(
        lattice_constants=[1.0, 1.0, 10.0],
        num_grids=[2, 2, 10],
        charge=1.0, sigma=1.0, defect_z_pos=0.0,
        epsilon=[[1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 2.5, 4.0, 4.0, 4.0, 2.5, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 3.5, 6.0, 6.0, 6.0, 3.5, 1.0, 1.0]],
        charge_profile=np.array([[charge]*2]*2),
        potential_profile=np.array([[potential]*3]*3))


def test_plot_profile(slab_gauss_model):
    slab_gauss_model.fp_xy_ave_potential = [-1.5, 1.5, 2.5, 4.5, 2.5, 1.5, -1.5, -2.5, -3.5, -1.5]
    plotter = ProfilePlotter(slab_gauss_model)
    plotter.plt.show()