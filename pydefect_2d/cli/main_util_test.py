# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace
from pathlib import Path

from pydefect.input_maker.supercell_info import SupercellInfo

from pydefect_2d.cli.main_util import parse_args_main_util_vasp
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist


def test_plot_volumetric_data():
    parsed_args = parse_args_main_util_vasp(["pvd", "-f", "LOCPOT"])
    expected = Namespace(filename="LOCPOT", direction=2, func=parsed_args.func)

    assert parsed_args == expected


def test_gauss_model_from_z(mocker):
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_const_dist.json":
            return mock_dielectric_dist
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.loadfn", side_effect=side_effect)

    mock_supercell_info = mocker.Mock(spec=SupercellInfo, autospec=True)

    def side_effect_si(filename):
        if filename == "supercell_info.json":
            return mock_supercell_info
        else:
            raise ValueError

    mocker.patch("pydefect.cli.main.loadfn", side_effect=side_effect_si)

    parsed_args = parse_args_main_util_vasp([
        "gmz",
        "-z", "0.1", "0.2",
        "-dd", "dielectric_const_dist.json",
        "-cd", "correction"])

    expected = Namespace(supercell_info=mock_supercell_info,
                         z_pos=[0.1, 0.2],
                         diele_dist=mock_dielectric_dist,
                         std_dev=0.5,
                         multiprocess=True,
                         k_max=5.0,
                         k_mesh_dist=0.05,
                         correction_dir=Path("correction"),
                         func=parsed_args.func,
                         )

    assert parsed_args == expected


