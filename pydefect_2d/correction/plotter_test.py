# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest

from pydefect_2d.correction.plotter import ProfilePlotter
from pydefect_2d.correction.slab_model_info import SlabGaussModel


@pytest.fixture
def slab_gauss_model():
    charge = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    potential = [-1.0, 1.0, 2.0, 4.0, 2.0, 1.0, -1.0, -2.0, -3.0, -2.0]
    return SlabGaussModel(
        charge=1.0, std_dev=1.0, defect_z_pos=0.0,
        lattice_constants=[1.0, 1.0, 10.0],
        epsilon=[[1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 2.5, 4.0, 4.0, 4.0, 2.5, 1.0, 1.0],
                 [1.0, 1.0, 1.0, 3.5, 6.0, 6.0, 6.0, 3.5, 1.0, 1.0]],
        charge_profile=np.array([[charge]*2]*2),
        potential_profile=np.array([[potential]*3]*3))


def test_plot_profile(slab_gauss_model):
    plotter = ProfilePlotter(slab_gauss_model)
    plotter.plt.show()