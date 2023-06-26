# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import numpy as np
import pytest
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.potential.epsilon_distribution import \
    EpsilonGaussianDistribution
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.slab_model_info import CalcGaussChargePotential, \
    GaussChargeModel, SlabModel, FP1dPotential

grid = Grid(10., 4)


@pytest.fixture(scope="session")
def epsilon_dist():
    return EpsilonGaussianDistribution(
        grid=grid,
        ave_electronic_epsilon=[0.5]*3,
        ave_ionic_epsilon=[0.]*3,
        center=5.0, sigma=0.1)


@pytest.fixture(scope="session")
def gauss_model():
    return GaussChargeModel(Grids([grid, grid, grid]),
                            sigma=1.0,
                            defect_z_pos=0.0,
                            epsilon_x=np.array([1.0]*4),
                            epsilon_y=np.array([1.0]*4))


@pytest.fixture(scope="session")
def potential(epsilon_dist, gauss_model):
    return CalcGaussChargePotential(epsilon_dist, gauss_model).potential


@pytest.fixture(scope="session")
def fp_1d_potential():
    return FP1dPotential(grid=grid, potential=[0.1, 0.2, 0.3, 0.4])


@pytest.fixture(scope="session")
def slab_model(epsilon_dist, gauss_model, potential, fp_1d_potential):
    return SlabModel(1, epsilon_dist, gauss_model, potential, fp_1d_potential)


def test_json_file_mixin(gauss_model, potential, fp_1d_potential, tmpdir):
    assert_json_roundtrip(gauss_model, tmpdir)
    assert_json_roundtrip(potential, tmpdir)
    assert_json_roundtrip(fp_1d_potential, tmpdir)


def test_gauss_charge_model_charges(gauss_model: GaussChargeModel):
    assert gauss_model.charges[0][0][0] == 0.06349363593424097
    assert gauss_model.farthest_z_from_defect == (2, 5.0)

