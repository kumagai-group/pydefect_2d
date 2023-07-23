# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace

from pydefect.analyzer.defect_structure_info import DefectStructureInfo

from pydefect_2d.potential.dielectric_distribution import DielectricConstDist
from pydefect_2d.potential.slab_model_info import GaussChargeModel
from pydefect_2d.vasp.cli.main import parse_args_main_vasp


def test_make_dielectric_distribution(mocker):
    mock_unitcell = mocker.patch("pydefect_2d.vasp.cli.main.Unitcell")
    mock_structure = mocker.patch("pydefect_2d.vasp.cli.main.Structure")

    parsed_args = parse_args_main_vasp(["ed",
                                        "-u", "unitcell.yaml",
                                        "-s", "CONTCAR",
                                        "-n", "100",
                                        "-t", "gauss",
                                        "-p", "0.5",
                                        "--sigma", "0.1",
                                        "-sl", "0.2",
                                        "-sr", "0.3",
                                        "--error_func_width", "0.4"])
    expected = Namespace(
        unitcell=mock_unitcell.from_yaml.return_value,
        structure=mock_structure.from_file.return_value,
        num_grid=100,
        type="gauss",
        position=0.5,
        sigma=0.1,
        step_left=0.2,
        step_right=0.3,
        error_func_width=0.4,
        func=parsed_args.func)

    assert parsed_args == expected
    mock_unitcell.from_yaml.assert_called_once_with("unitcell.yaml")
    mock_structure.from_file.assert_called_once_with("CONTCAR")


def test_make_gauss_charge_models(mocker):
    mock_defect_structure = mocker.Mock(spec=DefectStructureInfo, autospec=True)
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)

    def side_effect(filename):
        if filename == "defect_structure_info.json":
            return mock_defect_structure
        elif filename == "dielectric_distribution.json":
            return mock_dielectric_dist
        else:
            raise ValueError

    mocker.patch("pydefect_2d.vasp.cli.main.loadfn", side_effect=side_effect)

    parsed_args = parse_args_main_vasp(["gcm",
                                        "-dsi", "defect_structure_info.json",
                                        "-d", "dielectric_distribution.json",
                                        "--sigma", "0.1"])
    expected = Namespace(
        defect_structure_info=mock_defect_structure,
        dielectric_dist=mock_dielectric_dist,
        sigma=0.1,
        func=parsed_args.func)

    assert parsed_args == expected


def test_calc_potential(mocker):
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)
    mock_gauss_model = mocker.Mock(spec=GaussChargeModel, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_distribution.json":
            return mock_dielectric_dist
        elif filename == "gauss_charge_model.json":
            return mock_gauss_model
        else:
            raise ValueError

    mocker.patch("pydefect_2d.vasp.cli.main.loadfn", side_effect=side_effect)

    parsed_args = parse_args_main_vasp(["gcp",
                                        "-d", "dielectric_distribution.json",
                                        "-g", "gauss_charge_model.json",
                                        "--no_multiprocess"])
    expected = Namespace(
        dielectric_dist=mock_dielectric_dist,
        gauss_charge_model=mock_gauss_model,
        multiprocess=False,
        func=parsed_args.func)

    assert parsed_args == expected


def test_make_fp_1d_potential(mocker):
    mock_locpot_defect = mocker.Mock()
    mock_locpot_perfect = mocker.Mock()

    def side_effect_locpot(filename):
        if filename == "LOCPOT_defect":
            return mock_locpot_defect
        elif filename == "LOCPOT_perfect":
            return mock_locpot_perfect
        else:
            raise ValueError

    mocker.patch("pydefect_2d.vasp.cli.main.Locpot.from_file",
                 side_effect=side_effect_locpot)

    parsed_args = parse_args_main_vasp(["fp",
                                        "-dl", "LOCPOT_defect",
                                        "-pl", "LOCPOT_perfect",
                                        "-a", "1"])
    expected = Namespace(
        defect_locpot=mock_locpot_defect,
        perfect_locpot=mock_locpot_perfect,
        axis=1,
        func=parsed_args.func)

    assert parsed_args == expected

