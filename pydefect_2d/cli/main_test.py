# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace
from pathlib import Path

from pydefect.analyzer.calc_results import CalcResults
from pydefect.input_maker.supercell_info import SupercellInfo

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.potential.slab_model_info import GaussChargeModel
from pydefect_2d.cli.main import parse_args_main_vasp


def test_gauss_diele_dist(mocker):
    mock_unitcell = mocker.patch("pydefect.cli.main.Unitcell")
    mock_structure = mocker.patch("pydefect_2d.cli.main.Structure")

    parsed_args = parse_args_main_vasp(["gdd",
                                        "-u", "unitcell.yaml",
                                        "-p", "CONTCAR",
                                        "-c", "0.5",
                                        "--std_dev", "0.1"])
    expected = Namespace(
        unitcell=mock_unitcell.from_yaml.return_value,
        perfect_slab=mock_structure.from_file.return_value,
        mesh_distance=0.05,
        center=0.5,
        std_dev=0.1,
        func=parsed_args.func)

    assert parsed_args == expected
    mock_unitcell.from_yaml.assert_called_once_with("unitcell.yaml")
    mock_structure.from_file.assert_called_once_with("CONTCAR")


def test_step_diele_dist(mocker):
    mock_unitcell = mocker.patch("pydefect.cli.main.Unitcell")
    mock_structure = mocker.patch("pydefect_2d.cli.main.Structure")

    parsed_args = parse_args_main_vasp(["sdd",
                                        "-u", "unitcell.yaml",
                                        "-p", "CONTCAR",
                                        "-c", "0.5",
                                        "-w", "0.1"])
    expected = Namespace(
        unitcell=mock_unitcell.from_yaml.return_value,
        perfect_slab=mock_structure.from_file.return_value,
        mesh_distance=0.05,
        center=0.5,
        step_width=0.1,
        error_func_width=0.3,
        func=parsed_args.func)

    assert parsed_args == expected
    mock_unitcell.from_yaml.assert_called_once_with("unitcell.yaml")
    mock_structure.from_file.assert_called_once_with("CONTCAR")


def test_make_1d_gauss_models(mocker):
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_distribution.json":
            return mock_dielectric_dist
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.loadfn", side_effect=side_effect)

    mock_supercell_info = mocker.Mock(spec=SupercellInfo, autospec=True)

    def side_effect_si(filename):
        if filename == "dielectric_distribution.json":
            return mock_dielectric_dist
        elif filename == "supercell_info.json":
            return mock_supercell_info
        else:
            raise ValueError

    mocker.patch("pydefect.cli.main.loadfn", side_effect=side_effect_si)

    mock_locpot_perfect = mocker.Mock()

    def side_effect_locpot(filename):
        if filename == "LOCPOT_perfect":
            return mock_locpot_perfect
        else:
            raise ValueError
    mocker.patch("pydefect_2d.cli.main.Locpot.from_file",
                 side_effect=side_effect_locpot)

    parsed_args = parse_args_main_vasp(["1gm",
                                        "-dd", "dielectric_distribution.json",
                                        "-r", "0.1", "0.2",
                                        "-pl", "LOCPOT_perfect",
                                        "-s", "supercell_info.json"])
    expected = Namespace(
        diele_dist=mock_dielectric_dist,
        range=[0.1, 0.2],
        supercell_info=mock_supercell_info,
        perfect_locpot=mock_locpot_perfect,
        std_dev=0.5,
        func=parsed_args.func)

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

    parsed_args = parse_args_main_vasp(["fp",
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

    parsed_args = parse_args_main_vasp(["gm",
                                        "-dd", "dielectric_const_dist.json",
                                        "-d", "Va_O1_1",
                                        ])
    expected = Namespace(
        diele_dist=mock_dielectric_dist,
        std_dev=0.5,
        multiprocess=True,
        k_max=5.0,
        k_mesh_dist=0.05,
        dirs=[Path("Va_O1_1")],
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

    parsed_args = parse_args_main_vasp(["sm",
                                        "-dd", "dielectric_distribution.json",
                                        "-pcr", "perfect_calc_results.json",
                                        "-d", "Va_O1_0",
                                        "-cd", "correction"])
    expected = Namespace(
        diele_dist=mock_dielectric_dist,
        perfect_calc_results=mock_perfect_calc_results,
        dirs=[Path("Va_O1_0")],
        correction_dir=Path("correction"),
        func=parsed_args.func)

    assert parsed_args == expected
