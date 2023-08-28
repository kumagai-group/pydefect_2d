# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest

from pydefect_2d.potential.dielectric_distribution import DielectricConstDist
from pydefect_2d.potential.distribution import StepDist
from pydefect_2d.potential.make_extended_diele_dist import extend_diele_dist, \
    new_perp_diele


@pytest.fixture
def diele_dist():
    return DielectricConstDist(ave_ele=[2.0]*3, ave_ion=[1.0]*3,
                               dist=StepDist(length=10.0,
                                             num_grid=10,
                                             step_left=2.5,
                                             step_right=7.5,
                                             error_func_width=1.0))


def test_extended_diele_dist(diele_dist):
    actual = extend_diele_dist(diele_dist, 2)
    expected = DielectricConstDist(ave_ele=[1.5, 1.5, 1.3333333333333333],
                                   ave_ion=[0.5, 0.5, 0.16666666666666674],
                                   dist=StepDist(length=20.0,
                                                 num_grid=20,
                                                 step_left=2.5,
                                                 step_right=7.5,
                                                 error_func_width=1.0))
    assert actual == expected


def test_new_perp_diele():
    assert new_perp_diele(2.0, mul=2) == 2 * 2.0 / (1 + 2.0)

