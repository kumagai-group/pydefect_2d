# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import pytest
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.potential.make_epsilon_distribution import \
    EpsilonGaussianDistribution
from pydefect_2d.potential.grids import Grid, Grids
from pydefect_2d.potential.slab_model_info import CalcPotential, \
    GaussChargeModel, SlabModel, GaussElectrostaticEnergy, FP1dPotential

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
def gauss_electrostatic_energy():
    return GaussElectrostaticEnergy(electrostatic_energy=1., charge=1)


@pytest.fixture(scope="session")
def fp_1d_potential():
    return FP1dPotential(grid=grid, potential=[0.1, 0.2, 0.3, 0.4])


@pytest.fixture(scope="session")
def slab_model(epsilon_dist, gauss_model, potential, fp_1d_potential):
    return SlabModel(epsilon_dist, gauss_model, potential, fp_1d_potential)


def test_json_file_mixin(gauss_model, potential, gauss_electrostatic_energy,
                         fp_1d_potential, tmpdir):
    assert_json_roundtrip(gauss_model, tmpdir)
    assert_json_roundtrip(potential, tmpdir)
    assert_json_roundtrip(gauss_electrostatic_energy, tmpdir)
    assert_json_roundtrip(fp_1d_potential, tmpdir)


def test_gauss_charge_model_charges(gauss_model: GaussChargeModel):
    assert gauss_model.charges[0][0][0] == 0.06349363593424097
    assert gauss_model.farthest_z_from_defect == (2, 5.0)


def test_slab_gauss_model_electrostatic_energy(slab_model):
    energy = 4.827933538413449
    expected = GaussElectrostaticEnergy(electrostatic_energy=energy,
                                        charge=1,
                                        mul=1,
                                        alignment=0.964442479030666)
    assert slab_model.to_electrostatic_energy == expected




