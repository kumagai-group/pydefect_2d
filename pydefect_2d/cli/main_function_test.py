# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pathlib import Path

from pydefect_2d.cli.main import parse_args_main_vasp
from pydefect_2d.cli.main_function import make_gauss_diele_dist, \
    make_step_diele_dist, make_1d_gauss_models, make_gauss_model_from_z, \
    make_1d_fp_potential, make_1d_slab_model


def test_make_gauss_diele_dist(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["gdd",
         "--unitcell", str(test_files / "unitcell.yaml"),
         "-pl", str(test_files / "perfect" / "LOCPOT"),
         "--center", "0.5",
         "--std_dev", "1.0",
         "--denominator", "4"])
    make_gauss_diele_dist(args)


def test_make_gausses_diele_dist(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["gdd",
         "--unitcell", str(test_files / "unitcell.yaml"),
         "-pl", str(test_files / "perfect" / "LOCPOT"),
         "--center", "0.2", "0.4",
         "--std_dev", "1.0",
         "--denominator", "4"])
    make_gauss_diele_dist(args)


def test_make_step_diele_dist(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["sdd",
         "--unitcell", str(test_files / "unitcell.yaml"),
         "-pl", str(test_files / "perfect" / "LOCPOT"),
         "--center", "0.5",
         "--step_width", "3.0",
         "--step_width_z", "3.4",
         "--std_dev", "0.3",
         "--denominator", "4"])
    make_step_diele_dist(args)


def test_make_1d_gauss_models(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    Path("gauss_1d_potential_0.000.json").touch()
    args = parse_args_main_vasp([
        "1gm",
        "-dd", str(test_files / "dielectric_const_dist.json"),
        "-s", str(test_files / "supercell_info.json"),
        "-r", "0.2", "-0.2",
        "--std_dev", "0.5",
        "-m", "0.05",
        ])
    make_1d_gauss_models(args)


def test_gauss_model_from_z(test_files, tmpdir):
    print(tmpdir)

    args = parse_args_main_vasp(
        ["gmz",
         "-s", str(test_files / "supercell_info.json"),
         "-z", "0.5",
         "-dd", str(test_files / "dielectric_const_dist.json"),
         "--std_dev", "0.5",
         "--no_multiprocess",
         "--k_max", "1.0",
         "--k_mesh_dist", "0.5",
         "-cd", str(tmpdir),
         ])

#    file = args.correction_dir / "gauss_charge_model_0.500.json"
#    file.unlink(missing_ok=True)
    make_gauss_model_from_z(args)


def test_make_1d_fp_potential(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["1fp",
         "-d", str(test_files / "H_ad_1"),
         "-pl", str(test_files / "perfect" / "LOCPOT"),
         "-od", str(test_files / "1d_gauss"),
         ])
    make_1d_fp_potential(args)


def test_make_1d_slab_model(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_vasp(
        ["1sm",
         "-d", str(test_files / "H_ad_1"),
         "-dd", str(test_files / "dielectric_const_dist.json"),
         "-od", str(test_files / "1d_gauss"),
         "-g", str(test_files / "correction" / "gauss_energies.json"),
         ])
    make_1d_slab_model(args)
