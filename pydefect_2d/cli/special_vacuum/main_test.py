# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from argparse import Namespace
from pathlib import Path

from pydefect_2d.cli.special_vacuum.main import parse_args_special_vac
from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.three_d.slab_model import GaussChargeModel


def test_special_vacuum_extend_dielectric_dist(mocker):

    mock_dielectric_dist = mocker.Mock(spec=DielectricConstDist, autospec=True)
    mock_gauss_charge_model = mocker.Mock(spec=GaussChargeModel, autospec=True)

    def side_effect(filename):
        if filename == "dielectric_const_dist.json":
            return mock_dielectric_dist
        elif filename == "gauss_charge_model.json":
            return mock_gauss_charge_model
        else:
            raise ValueError

    mocker.patch("pydefect_2d.cli.main.loadfn", side_effect=side_effect)

    parsed_args = parse_args_special_vac(["edd",
                                          "-l", "20", "30",
                                          "-dd", "dielectric_const_dist.json",
                                          "-gcm", "gauss_charge_model.json",
                                          ])
    expected = Namespace(
        slab_lengths=[20.0, 30.0],
        diele_dist=mock_dielectric_dist,
        gauss_charge_model=mock_gauss_charge_model,
        func=parsed_args.func)

    assert parsed_args == expected


def test_special_vacuum():

    parsed_args = parse_args_special_vac(["sv",
                                          "-d", "a", "b"])
    expected = Namespace(
        dirs=[Path("a"), Path("b")],
        func=parsed_args.func)

    assert parsed_args == expected
