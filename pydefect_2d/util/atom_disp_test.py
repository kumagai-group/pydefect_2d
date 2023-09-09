# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pathlib import Path

from pymatgen.core import Structure

from pydefect_2d.util.atom_disp import atom_disp, disp_from_yaml


def test_atom_disp():
    s = Structure.from_str("""H
    1.000000000000000
    2.0 0.0 0.0
    0.0 2.0 0.0
    0.0 0.0 2.0
   H   
     2
Direct
0 0 0
0 0 0.5""", fmt="poscar")

#    atom_disp_from_yaml()
    actual = atom_disp(s, {0: [0.0, 0.0, 0.2]})
    expected = Structure.from_str("""H
    1.000000000000000
    2.0 0.0 0.0
    0.0 2.0 0.0
    0.0 0.0 2.0
   H   
     2
Direct
0 0 0.1
0 0 0.5""", fmt="poscar")

    assert actual == expected


def test_disp_from_yaml(tmpdir):
    tmpdir.chdir()
    file_name = "atom_disp.yaml"
    Path(file_name).write_text("40: 0. 0. 0.2")
    assert disp_from_yaml(file_name) == {40: [0., 0., 0.2]}
