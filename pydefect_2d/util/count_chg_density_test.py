# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
import numpy as np
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.vasp import Chgcar

from pydefect_2d.util.count_chg_density import count_chg_density, vacuum_range


def test_count_chg_density():
    structure = Structure.from_str("""test
1.0
2.0 0.0 0.0
0.0 2.0 0.0
0.0 0.0 20.0
   H
     1
Direct
0.0 0.0  0.5
""", fmt="poscar")
    chgcar = Chgcar(structure, data={"total": np.array([[[1.0]*10]])})
    actual = count_chg_density(chgcar, (0.46, 0.56))
    assert actual == 0.1

    actual = count_chg_density(chgcar, (0.56, 0.46))
    assert actual == 0.9


def test_vacuum_range(test_files):
    diele_const = loadfn(test_files / "dielectric_const_dist.json")
    actual = vacuum_range(diele_const, thr=2.5)
    assert actual == (0.5833333333333333, 0.4444444444444444)
