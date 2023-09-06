# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from pydefect_2d.cli.main_util import parse_args_main_util_vasp
from pydefect_2d.cli.main_util_function import plot_volumetric_data, \
    make_gauss_model_from_z


def test_plot_volumetric_data(test_files, tmpdir):
    print(tmpdir)
    tmpdir.chdir()
    parsed_args = parse_args_main_util_vasp([
        "pvd", "-f", str(test_files / "perfect" / "LOCPOT")])
    plot_volumetric_data(parsed_args)


def test_gauss_model_from_z(test_files, tmpdir):
    print(tmpdir)

    args = parse_args_main_util_vasp(
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
