# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace

import numpy as np
from numpy import linspace
from pymatgen.core import Structure

from pydefect_2d.potential.make_epsilon_distribution import EpsilonDistribution
from pydefect_2d.potential.grids import Grid
from pydefect_2d.vasp.cli.main_function import \
    make_epsilon_distributions


def test_make_epsilon_distributions(mocker):
    unitcell = mocker.MagicMock(ele_dielectric_const=list(np.eye(3)*2),
                                ion_dielectric_const=list(np.eye(3)*3))
    unitcell_structure = Structure.from_str("""B1 N1
    1.000000000000000
     2.5049600544457160    0.0000000000000000    0.0000000000000000
    -1.2524800272228580    2.1693590426340372    0.0000000000000000
     0.0000000000000000    0.0000000000000000   10.0000000000000000
   B    N
     1     1
Direct
  0.0000000000000000  0.0000000000000000  0.5000000000000000
  0.3333333332999970  0.6666666667000030  0.5000000000000000""", fmt="poscar")
    args = Namespace(unitcell=unitcell,
                     structure=unitcell_structure,
                     position=0.5,
                     num_grid=10,
                     sigma=0.5,
                     muls=[1, 2])

    actual = make_epsilon_distributions(args)
    expected = EpsilonDistribution(Grid(10.0, 10),
                                   electronic=[[]],
                                   ionic=[[]])
    print(actual)


