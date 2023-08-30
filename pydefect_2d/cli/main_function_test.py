# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

from pydefect_2d.cli.main import parse_args_main_vasp
from pydefect_2d.cli.main_function import make_gauss_diele_dist


def test_make_gauss_diele_dist(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["gdd",
         "--unitcell", str(test_files / "main_function" / "unitcell.yaml"),
         "--perfect_slab", str(test_files / "main_function" / "POSCAR_slab"),
         "--mesh_distance", "0.1",
         "--center", "0.5",
         "--sigma", "1.0"])
    make_gauss_diele_dist(args)


