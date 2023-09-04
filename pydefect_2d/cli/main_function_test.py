# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pathlib import Path

from pydefect_2d.cli.main import parse_args_main_vasp
from pydefect_2d.cli.main_function import make_gauss_diele_dist, \
    make_step_diele_dist, make_1d_gauss_models, make_fp_1d_potential, \
    make_gauss_model, make_slab_model


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


def test_make_step_diele_dist(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["sdd",
         "--unitcell", str(test_files / "main_function" / "unitcell.yaml"),
         "--perfect_slab", str(test_files / "main_function" / "POSCAR_slab"),
         "--mesh_distance", "0.1",
         "--center", "0.5",
         "--step_width", "3.0",
         "--error_func_width", "0.3"])
    make_step_diele_dist(args)


def test_make_1d_gauss_models(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    Path("gauss1_d_potential_0.000.json").touch()
    args = parse_args_main_vasp(
        ["1gm",
         "-dd", str(test_files / "main_function" / "dielectric_const_dist.json"),
         "-r", "0.2", "-0.2",
         "-si", str(test_files / "main_function" / "supercell_info.json"),
         "--sigma", "0.5",
         "-pl", str(test_files / "main_function" / "perfect" / "LOCPOT"),
         ])
    make_1d_gauss_models(args)


def test_make_fp_1d_potential(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["fp",
         "-d", str(test_files / "main_function" / "H_ad_1"),
         "-pl", str(test_files / "main_function" / "perfect" / "LOCPOT"),
         "-p", str(test_files / "main_function" / "1d_pots"),
         ])
    make_fp_1d_potential(args)


def test_make_gauss_model(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["gm",
         "-dd", str(test_files / "main_function" / "dielectric_const_dist.json"),
         "--sigma", "0.5",
         "--no_multiprocess",
         "--k_max", "1.0",
         "--k_mesh_dist", "0.5",
         "-d", str(test_files / "main_function" / "H_ad_1/"),
         ])
    make_gauss_model(args)


def test_make_slab_model(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["sm",
         "-dd", str(test_files / "main_function" / "dielectric_const_dist.json"),
         "-pcr", str(test_files / "main_function" / "perfect" / "calc_results.json"),
         "-d", str(test_files / "main_function" / "H_ad_1/"),
         "-cd", str(test_files / "main_function" / "correction")])
    make_slab_model(args)

