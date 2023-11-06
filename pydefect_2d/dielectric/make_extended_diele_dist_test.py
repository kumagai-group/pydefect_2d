# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.dielectric.distribution import StepDist
from pydefect_2d.dielectric.make_extended_diele_dist import AddVacuum, \
    RepeatDieleDist


@pytest.fixture
def diele_dist():
    dist = StepDist(length=10.0,
                    num_grid=10,
                    center=5.0,
                    in_plane_width=5.0,
                    in_plane_sigma=1.0,
                    out_of_plane_width=5.0,
                    out_of_plane_sigma=1.0)
    return DielectricConstDist(ave_ele=[2.0]*3, ave_ion=[1.0]*3, dist=dist)


def test_add_vacuum(diele_dist):
    actual = AddVacuum(diele_dist, 10.0).diele_const_dist
    dist = StepDist(length=20.0,
                    num_grid=20,
                    center=10.0,
                    in_plane_width=5.0,
                    in_plane_sigma=1.0,
                    out_of_plane_width=5.0,
                    out_of_plane_sigma=1.0)
    expected = DielectricConstDist(ave_ele=[1.5, 1.5, 1.3333333333333333],
                                   ave_ion=[0.5, 0.5, 0.16666666666666674],
                                   dist=dist)
    assert actual == expected


def test_repeat_diele_dist(diele_dist):
    actual = RepeatDieleDist(diele_dist, 2).diele_const_dist
    print(actual)