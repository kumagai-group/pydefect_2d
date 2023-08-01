# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import numpy as np
from matplotlib import pyplot as plt

from pydefect_2d.potential.calc_one_d_potential import OneDGaussChargeModel, \
    Calc1DPotential
from pydefect_2d.potential.dielectric_distribution import DielectricConstDist
from pydefect_2d.potential.distribution import ManualDist
from pydefect_2d.potential.grids import Grid


def test_calc_1d_potential():
    n_grid = 100
    grid = Grid(10, n_grid)
    dist = ManualDist.from_grid(grid, np.array([1.0]*n_grid))
    charge_model = OneDGaussChargeModel(grid=grid,
                                        surface=100.,
                                        sigma=0.1,
                                        defect_z_pos_in_frac=0.5)
    diele_dist = DielectricConstDist([1.]*3, [0.]*3, dist)

    calc_1_potential = Calc1DPotential(diele_dist, charge_model)
    calc_1_potential.potential.to_plot(plt.gca())
    plt.show()
