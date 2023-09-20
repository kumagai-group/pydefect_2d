# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace
from pathlib import Path

from pydefect.analyzer.calc_results import CalcResults
from pydefect.input_maker.supercell_info import SupercellInfo

from pydefect_2d.cli.main_util import parse_args_main_util_vasp
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist


def test_plot_volumetric_data():
    parsed_args = parse_args_main_util_vasp([
        "pvd",
        "-f", "LOCPOT",
        "-y", "0.0", "1.0"])
    expected = Namespace(filename="LOCPOT",
                         direction=2,
                         y_range=[0.0, 1.0],
                         target_val=None,
                         z_guess=None,
                         func=parsed_args.func)

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


def test_fp_1d_potential(mocker):
    mock_locpot_perfect = mocker.Mock()

    def side_effect_locpot(filename):
        if filename == "LOCPOT_perfect":
            return mock_locpot_perfect
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.Locpot.from_file",
                 side_effect=side_effect_locpot)

    parsed_args = parse_args_main_util_vasp(["fp",
                                             "-d", "Va_O1_1",
                                             "-pl", "LOCPOT_perfect",
                                             "-p", "1d_pots"])
    expected = Namespace(
        dirs=[Path("Va_O1_1")],
        perfect_locpot=mock_locpot_perfect,
        pot_dir=Path("1d_pots"),
        func=parsed_args.func)

    assert parsed_args == expected


def test_make_gauss_model(mocker):
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_const_dist.json":
            return mock_dielectric_dist
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.loadfn", side_effect=side_effect)

    parsed_args = parse_args_main_util_vasp(["gm",
                                             "-dd",
                                             "dielectric_const_dist.json",
                                             "-d", "Va_O1_1",
                                             "-cd", "correction"])
    expected = Namespace(
        diele_dist=mock_dielectric_dist,
        std_dev=0.5,
        multiprocess=True,
        k_max=5.0,
        k_mesh_dist=0.05,
        dirs=[Path("Va_O1_1")],
        correction_dir=Path("correction"),
        func=parsed_args.func)

    assert parsed_args == expected


def test_slab_model(mocker):
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)
    mock_perfect_calc_results = mocker.Mock(spec=CalcResults, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_distribution.json":
            return mock_dielectric_dist
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.loadfn", side_effect=side_effect)

    def side_effect_2(filename):
        if filename == "perfect_calc_results.json":
            return mock_perfect_calc_results
        else:
            raise ValueError

    mocker.patch("pydefect.cli.main.loadfn", side_effect=side_effect_2)

    parsed_args = parse_args_main_util_vasp(["sm",
                                             "-dd",
                                             "dielectric_distribution.json",
                                             "-pcr",
                                             "perfect_calc_results.json",
                                             "-d", "Va_O1_0",
                                             "-cd", "correction"])
    expected = Namespace(
        diele_dist=mock_dielectric_dist,
        perfect_calc_results=mock_perfect_calc_results,
        dirs=[Path("Va_O1_0")],
        correction_dir=Path("correction"),
        func=parsed_args.func)

    assert parsed_args == expected


