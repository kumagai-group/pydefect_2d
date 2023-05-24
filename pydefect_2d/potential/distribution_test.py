# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from numpy import linspace

from pydefect_2d.potential.distribution import \
    make_gaussian_distribution, make_step_like_distribution, \
    rescale_distribution


def test_make_step_like_epsilon():
    actual = make_step_like_distribution(
        grid=list(linspace(0, 10.0, 10, endpoint=False)),
        step_left=3.2, step_right=5.5, error_func_width=0.1)
    assert actual[3] == 0.0023388674905235884


def test_make_gaussian_epsilon():
    grid = list(linspace(0.0, 10.0, 10, endpoint=False))
    actual = make_gaussian_distribution(grid=grid,
                                        position=2.0,
                                        sigma=1.0)
    assert actual[1] == 0.6065306597126334


def test_rescale_distribution_ionic():
    actual = rescale_distribution(dist=[1.0, 3.0], average=1.0,
                                  is_ionic=True)
    expected = [0.5, 1.5]
    assert actual == expected


def test_rescale_distribution_ion_clamped():
    actual = rescale_distribution(dist=[1.0, 3.0], average=1.2,
                                  is_ionic=False)
    expected = [1.0, 1.4]
    assert actual == expected
