# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from math import ceil
from typing import List

import numpy as np
import pytest
from monty.json import MSONable
from numpy import pi
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from vise.tests.helpers.assertion import assert_json_roundtrip

from pydefect_2d.correction.slab_model_info import SlabGaussModel
from pydefect_2d.correction.matsurf import SlabModel


@pytest.fixture
def slab_gauss_model():
    return SlabGaussModel(lattice_constants=[10.0]*3,
                          epsilon=[[1.0, 2.0, 2.0, 1.0],
                                   [1.0, 2.0, 2.0, 1.0],
                                   [1.0, 2.0, 2.0, 1.0]],
                          charge=1.0, std_dev=1.0, defect_z_pos=0.0)


def test_json_file_mixin(slab_gauss_model, tmpdir):
    print(slab_gauss_model.real_charge)
    assert_json_roundtrip(slab_gauss_model, tmpdir)


def test_slab_gauss_model_grids(slab_gauss_model):
    assert slab_gauss_model.num_grids == [4, 4, 4]
    assert_array_almost_equal(slab_gauss_model.grids, np.array([[0.0, 2.5, 5.0, 7.5],
                                                                [0.0, 2.5, 5.0, 7.5],
                                                                [0.0, 2.5, 5.0, 7.5]]))


def test_slab_gauss_model_Gs(slab_gauss_model):
    pi_over_lat = 2 * pi / 10
    expected = np.array([[0, pi_over_lat, 2*pi_over_lat, pi_over_lat]] * 3)
    assert_array_almost_equal(slab_gauss_model.Gs, expected)


def test_slab_gauss_model_reciprocal_epsilon(slab_gauss_model):
    assert_array_almost_equal(slab_gauss_model.reciprocal_epsilon,
                              [np.array([6.+0j, -1.-1.j, 0.+0j, -1.+1.j])] * 3)


def test_slab_gauss_model_volume(slab_gauss_model):
    assert slab_gauss_model.volume == 1000.0


def test_slab_gauss_model_electrostat_energy(slab_gauss_model: SlabGaussModel):
    assert slab_gauss_model.electrostatic_energy == 4.259041251566515





