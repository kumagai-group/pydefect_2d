# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

from pydefect_2d.util.utils import get_z_from_filename


def test_get_z_from_filename():
    assert get_z_from_filename("isolated_gauss_energy_0.370.json") == 0.37

