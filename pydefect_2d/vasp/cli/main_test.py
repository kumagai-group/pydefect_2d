# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace

from pydefect_2d.vasp.cli.main import parse_args_main_vasp


def test_unitcell(mocker):
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
