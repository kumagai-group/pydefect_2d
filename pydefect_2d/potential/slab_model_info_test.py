# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import pytest
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.potential.make_epsilon_distribution import \
    EpsilonGaussianDistribution
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.slab_model_info import CalcPotential, \
    GaussChargeModel, SlabModel

grid = Grid(10., 4)


@pytest.fixture(scope="session")
def epsilon_dist():
    return EpsilonGaussianDistribution(grid=grid,
                                       electronic=[[0., 1., 1., 0.]] * 3,
                                       ionic=[[0., 0., 0., 0.]] * 3,
                                       center=5.0, sigma=0.1)


@pytest.fixture(scope="session")
def gauss_model():
    return GaussChargeModel(Grids([grid, grid, grid]),
                            charge=1,
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




