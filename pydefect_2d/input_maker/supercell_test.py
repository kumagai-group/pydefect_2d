# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import pytest

from pydefect_2d.input_maker.supercell import Supercell


def test_supercell(single_BN, single_BN_2x2):
    actual = Supercell(single_BN, [[2, 0], [0, 2]]).structure
    expected = single_BN_2x2
    assert actual == expected


# def test_supercell_isotropy(single_BN, single_BN_2x2):
