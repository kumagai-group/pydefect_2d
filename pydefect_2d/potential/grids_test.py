# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import pytest

from pydefect_2d.potential.grids import Grid, Grids


@pytest.fixture
def grid():
    return Grid(base_length=1.0, base_num_grid=2, mul=2)


@pytest.fixture
def grids(grid):
    return Grids([grid]*3)


def test_grid_properties(grid):
    assert grid.length == 2.0
    assert grid.num_z_grid == 4
    assert grid.grid_points == [0.0, 0.5, 1.0, 1.5]


def test_grids_properties(grids, grid):
    assert grids() == [grid]*3
    assert grids.all_grid_points == [[0.0, 0.5, 1.0, 1.5]]*3
    assert grids.num_grid_points == [4, 4, 4]
    assert grids.lengths == [2.0, 2.0, 2.0]
    assert grids.xy_area == 4.0
    assert grids.z_length == 2.0
    assert grids.z_grid_points == [0.0, 0.5, 1.0, 1.5]
    assert grids.volume == 8.0
    assert grids.nearest_z_grid_point(0.4) == (1, 0.5)

