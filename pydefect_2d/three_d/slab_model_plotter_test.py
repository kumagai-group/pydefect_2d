# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
from matplotlib import pyplot as plt

from pydefect_2d.dielectric.distribution import ManualDist
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.three_d.grids import Grid, Grids
from pydefect_2d.three_d.slab_model_plotter import SlabModelPlotter
from pydefect_2d.three_d.slab_model import GaussChargePotential, \
    SlabModel, GaussChargeModel
from pydefect_2d.one_d.one_d_potential import Fp1DPotential


def test_slab_model_plotter():
    grid = Grid(10.0, 10)
    manual_dist = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    dist = ManualDist.from_grid(grid, manual_dist)
    diele_dist = DielectricConstDist(ave_ele=[3., 3., 0.5],
                                     ave_ion=[1., 1., 1.0], dist=dist)

    charges = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    grids = Grids.from_z_grid(np.array([[10., 0.], [0., 10.]]), grid)
    charge = GaussChargeModel(grids=grids,
                              std_dev=1.0,
                              gauss_pos_in_frac=0.0,
                              periodic_charges=np.array([[charges]*2]*2))

    pot = [-1.0, 1.0, 2.0, 4.0, 2.0, 1.0, -1.0, -2.0, -3.0, -2.0]
    potential = GaussChargePotential(
        grids=grids,
        potential=np.array([[pot]*2]*2))

    fp_pot_dist = np.array(
        [-1.5, 1.5, 2.5, 4.5, 2.5, 1.5, -1.5, -2.5, -3.5, -1.5])
    charge_state = 1
    fp_pot = Fp1DPotential(grid, fp_pot_dist)

    slab_model = SlabModel(diele_dist, charge, potential, charge_state, fp_pot)
    plotter = SlabModelPlotter(plt, slab_model)
    plotter.plt.show()

