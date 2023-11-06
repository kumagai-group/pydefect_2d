# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pydefect_2d.cli.main_util import parse_args_main_util_vasp
from pydefect_2d.cli.main_util_function import plot_volumetric_data, \
    make_gauss_model, \
    make_slab_model, add_vacuum, repeat_diele_dist


def test_plot_volumetric_data(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    parsed_args = parse_args_main_util_vasp([
        "pvd", "-f", str(test_files / "perfect" / "LOCPOT")])
    plot_volumetric_data(parsed_args)


def test_make_gauss_model(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_util_vasp(
        ["gm",
         "-dd", str(test_files / "dielectric_const_dist.json"),
         "--std_dev", "0.5",
         "--no_multiprocess",
         "--k_max", "1.0",
         "--k_mesh_dist", "0.5",
         "-d", str(test_files / "H_ad_1/"),
         "-cd", str(tmpdir),
         ])
    make_gauss_model(args)


def test_make_slab_model(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_util_vasp(
        ["sm",
         "-dd", str(test_files / "dielectric_const_dist.json"),
         "-pcr", str(test_files / "perfect" / "calc_results.json"),
         "-d", str(test_files / "H_ad_1/"),
         "-cd", str(test_files / "correction"),
         ])
    make_slab_model(args)


def test_add_vacuum(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_util_vasp(
        ["av",
         "-dd", str(test_files / "dielectric_const_dist.json"),
         "-l", "30"])
    add_vacuum(args)


def test_repeat_diele(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    args = parse_args_main_util_vasp(
        ["rd",
         "-dd", str(test_files / "dielectric_const_dist.json"),
         "-m", "2"])
    repeat_diele_dist(args)

