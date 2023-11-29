# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace
from pathlib import Path

from pydefect.input_maker.supercell_info import SupercellInfo

from pydefect_2d.correction.gauss_energy import GaussEnergies
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.cli.main import parse_args_main_vasp


def test_gauss_diele_dist(mocker):
    mock_unitcell = mocker.patch("pydefect.cli.main.Unitcell")

    mock_locpot_perfect = mocker.Mock()

    def side_effect_locpot(filename):
        if filename == "LOCPOT_perfect":
            return mock_locpot_perfect
        else:
            raise ValueError
    mocker.patch("pydefect_2d.cli.main.Locpot.from_file",
                 side_effect=side_effect_locpot)

    parsed_args = parse_args_main_vasp(["gdd",
                                        "-u", "unitcell.yaml",
                                        "-c", "0.5",
                                        "-pl", "LOCPOT_perfect",
                                        "--std_dev", "0.1",
                                        ])
    expected = Namespace(
        unitcell=mock_unitcell.from_yaml.return_value,
        center=[0.5],
        std_dev=0.1,
        denominator=1,
        perfect_locpot=mock_locpot_perfect,
        func=parsed_args.func)

    assert parsed_args == expected
    mock_unitcell.from_yaml.assert_called_once_with("unitcell.yaml")


def test_step_diele_dist(mocker):
    mock_unitcell = mocker.patch("pydefect.cli.main.Unitcell")

    mock_locpot_perfect = mocker.Mock()

    def side_effect_locpot(filename):
        if filename == "LOCPOT_perfect":
            return mock_locpot_perfect
        else:
            raise ValueError
    mocker.patch("pydefect_2d.cli.main.Locpot.from_file",
                 side_effect=side_effect_locpot)

    parsed_args = parse_args_main_vasp(["sdd",
                                        "-u", "unitcell.yaml",
                                        "-c", "0.5",
                                        "-w", "0.1",
                                        "-s", "0.5",
                                        "-pl", "LOCPOT_perfect"])
    expected = Namespace(
        unitcell=mock_unitcell.from_yaml.return_value,
        center=0.5,
        step_width=0.1,
        step_width_z=None,
        denominator=1,
        std_dev=0.5,
        perfect_locpot=mock_locpot_perfect,
        func=parsed_args.func)

    assert parsed_args == expected
    mock_unitcell.from_yaml.assert_called_once_with("unitcell.yaml")


def test_make_1d_gauss_models(mocker):
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

    parsed_args = parse_args_main_vasp(["1gm",
                                        "-dd", "dielectric_const_dist.json",
                                        "-r", "0.1", "0.2",
                                        "-s", "supercell_info.json"
                                        ])
    expected = Namespace(
        diele_dist=mock_dielectric_dist,
        range=[0.1, 0.2],
        supercell_info=mock_supercell_info,

        std_dev=0.5,
        mesh_distance=0.01,
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

    parsed_args = parse_args_main_vasp([
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


def test_make_1d_fp_potential(mocker):
    mock_locpot_perfect = mocker.Mock()

    def side_effect_locpot(filename):
        if filename == "LOCPOT_perfect":
            return mock_locpot_perfect
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.Locpot.from_file",
                 side_effect=side_effect_locpot)

    parsed_args = parse_args_main_vasp(["1fp",
                                        "-d", "Va_O1_1",
                                        "-pl", "LOCPOT_perfect",
                                        "-od", "1d_pots"])
    expected = Namespace(
        dirs=[Path("Va_O1_1")],
        perfect_locpot=mock_locpot_perfect,
        one_d_dir=Path("1d_pots"),
        func=parsed_args.func)

    assert parsed_args == expected


def test_make_1d_slab_model(mocker):
    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)
    mock_gauss_energies = mocker.Mock(spec=GaussEnergies, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_const_dist.json":
            return mock_dielectric_dist
        elif filename == "gauss_energies.json":
            return mock_gauss_energies
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.loadfn", side_effect=side_effect)

    parsed_args = parse_args_main_vasp(["1sm",
                                        "-d", "Va_O1_1",
                                        "-dd", "dielectric_const_dist.json",
                                        "-od", "1d_pots",
                                        "-g", "gauss_energies.json",
                                        "-s", "0.5"])
    expected = Namespace(
        dirs=[Path("Va_O1_1")],
        diele_dist=mock_dielectric_dist,
        one_d_dir=Path("1d_pots"),
        gauss_energies=mock_gauss_energies,
        slab_center=0.5,
        func=parsed_args.func)

    assert parsed_args == expected


