# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy import linspace
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.potential.make_epsilon_distribution import EpsilonDistribution, \
    Grid
from pydefect_2d.potential.slab_model_info import CalcPotential, \
    ProfilePlotter, GaussChargeModel, SlabModel, Potential, FP1dPotential

grid = Grid(10., 4)


@pytest.fixture(scope="session")
def epsilon_dist():
    return EpsilonDistribution(grid=grid,
                               electronic=[[0., 1., 1., 0.]] * 3,
                               ionic=[[0., 0., 0., 0.]] * 3,
                               center=5.0)


@pytest.fixture(scope="session")
def gauss_model():
    return GaussChargeModel([grid, grid, grid],
                            charge=1.0,
                            sigma=1.0,
                            defect_z_pos=0.0)


@pytest.fixture(scope="session")
def potential(epsilon_dist, gauss_model):
    return CalcPotential(epsilon_dist, gauss_model=gauss_model).potential


@pytest.fixture(scope="session")
def slab_model(epsilon_dist, gauss_model, potential):
    return SlabModel(epsilon_dist, gauss_model, potential)


def test_gauss_charge_model_charges(gauss_model):
    assert gauss_model.charges[0][0][0] == 0.06349363593424097


def test_json_file_mixin(gauss_model, potential, tmpdir):
    assert_json_roundtrip(gauss_model, tmpdir)
    assert_json_roundtrip(potential, tmpdir)


def test_slab_gauss_model_electrostatic_energy(slab_model):
    assert slab_model.electrostatic_energy == 4.827933538413449


def test_plot_profile():
    grid_plot = Grid(10.0, 10)
    epsilon = EpsilonDistribution(
        grid=grid_plot,
        electronic=[[1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 2.5, 4.0, 4.0, 4.0, 2.5, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 3.5, 6.0, 6.0, 6.0, 3.5, 1.0, 1.0]],
        ionic=[[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
               [0.0, 0.0, 0.0, 1.5, 3.0, 3.0, 3.0, 1.5, 0.0, 0.0],
               [0.0, 0.0, 0.0, 2.5, 5.0, 5.0, 5.0, 2.5, 0.0, 0.0]],
        center=0.5)
    grid_xy = Grid(1.0, 2)
    charges = [0.0, 1.0, 2.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    charge = GaussChargeModel(grids=[grid_xy, grid_xy, grid_plot],
                              charge=1.0, sigma=1.0, defect_z_pos=0.0,
                              charges=np.array([[charges]*2]*2))
    pot = [-1.0, 1.0, 2.0, 4.0, 2.0, 1.0, -1.0, -2.0, -3.0, -2.0]
    potential = Potential(
        grids=[grid_xy, grid_xy, grid_plot],
        potential=np.array([[pot]*2]*2))

    fp_pot = FP1dPotential(grid_plot, [-1.5, 1.5, 2.5, 4.5, 2.5, 1.5, -1.5, -2.5, -3.5, -1.5])

    slab_model = SlabModel(epsilon, charge, potential)
    plotter = ProfilePlotter(plt, slab_model, fp_pot)
    plotter.plt.show()




