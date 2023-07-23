# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from matplotlib import pyplot as plt

from pydefect_2d.potential.distribution import ManualDist
from pydefect_2d.potential.dielectric_distribution import DielectricConstDist
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.plotter import ProfilePlotter
from pydefect_2d.potential.slab_model_info import GaussChargePotential, \
    FP1dPotential, SlabModel, GaussChargeModel


def test_plot_profile():
    grid = Grid(10.0, 10)
    manual_dist = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    dist = ManualDist.from_grid(grid, manual_dist)
    diele_dist = DielectricConstDist(ave_ele=[3., 3., 0.5],
                                     ave_ion=[1., 1., 1.0], dist=dist)

    charges = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    grids = Grids.from_z_num_grid(np.array([[10., 0.], [0., 10.]]), grid)
    charge = GaussChargeModel(grids=grids,
                              sigma=1.0,
                              defect_z_pos_in_frac=0.0,
                              periodic_charges=np.array([[charges]*2]*2))

    pot = [-1.0, 1.0, 2.0, 4.0, 2.0, 1.0, -1.0, -2.0, -3.0, -2.0]
    potential = GaussChargePotential(
        grids=grids,
        potential=np.array([[pot]*2]*2))

    fp_pot_dist = [-1.5, 1.5, 2.5, 4.5, 2.5, 1.5, -1.5, -2.5, -3.5, -1.5]
    fp_pot = FP1dPotential(grid, fp_pot_dist)

    slab_model = SlabModel(diele_dist, charge, potential, 1, fp_pot)
    plotter = ProfilePlotter(plt, slab_model)
    plotter.plt.show()