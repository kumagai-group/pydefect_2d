# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace

from pydefect.input_maker.defect_entry import DefectEntry

from pydefect_2d.potential.make_epsilon_distribution import EpsilonDistribution
from pydefect_2d.vasp.cli.main import parse_args_main_vasp


def test_make_epsilon_distribution(mocker):
    mock_unitcell = mocker.patch("pydefect_2d.vasp.cli.main.Unitcell")
    mock_structure = mocker.patch("pydefect_2d.vasp.cli.main.Structure")

    parsed_args = parse_args_main_vasp(["med",
                                        "-u", "unitcell.yaml",
                                        "-s", "CONTCAR",
                                        "-p", "0.5",
                                        "-n", "100",
                                        "--sigma", "0.1"])
    expected = Namespace(
        unitcell=mock_unitcell.from_yaml.return_value,
        structure=mock_structure.from_file.return_value,
        position=0.5,
        num_grid=100,
        sigma=0.1,
        func=parsed_args.func)

    assert parsed_args == expected
    mock_unitcell.from_yaml.assert_called_once_with("unitcell.yaml")
    mock_structure.from_file.assert_called_once_with("CONTCAR")


def test_make_slab_gauss_model(mocker):
    mock_defect_entry = mocker.Mock(spec=DefectEntry, autospec=True)
    mock_epsilon_dist = mocker.Mock(spec=EpsilonDistribution, autospec=True)
    mocker_locpot = mocker.patch("pydefect_2d.vasp.cli.main.Locpot")

    def side_effect(filename):
        if filename == "defect_entry.json":
            return mock_defect_entry
        elif filename == "epsilon_distribution.json":
            return mock_epsilon_dist
        else:
            raise ValueError

    mocker.patch("pydefect_2d.vasp.cli.main.loadfn", side_effect=side_effect)

    parsed_args = parse_args_main_vasp(["msgm",
                                        "-d", "defect_entry.json",
                                        "-l", "LOCPOT",
                                        "-e", "epsilon_distribution.json",
                                        "--sigma", "0.1",
                                        "--no_potential_calc"])
    expected = Namespace(
        defect_entry=mock_defect_entry,
        locpot=mocker_locpot.from_file.return_value,
        epsilon_dist=mock_epsilon_dist,
        sigma=0.1,
        calc_potential=False,
        grid_divisor=10,
        func=parsed_args.func)

    assert parsed_args == expected

