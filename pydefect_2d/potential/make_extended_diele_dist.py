# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import argparse
import sys
from copy import deepcopy
from typing import List

import numpy as np
from monty.serialization import loadfn
from numpy import argmin
from numpy.testing import assert_almost_equal
from vise.util.logger import get_logger

from pydefect_2d.potential.dielectric_distribution import DielectricConstDist


logger = get_logger(__name__)


def extend_diele_dist_main(args):
    extend_diele_dist(args.diele_const_dist, args.muls)


def extend_diele_dist(diele_dist: DielectricConstDist, muls: List[float]):

    min_idx = argmin(diele_dist.dist.unscaled_dist)
    try:
        assert_almost_equal(diele_dist.dist.unscaled_dist[min_idx], 0., decimal=3)
    except AssertionError:
        logger.warning("Vacuum inserted position has finite distribution. You "
                       "should know what's happening.")

    for mul in muls:
        result = deepcopy(diele_dist)

        vac_len = (mul - 1.) * result.dist.length

        result.ave_ele = [1.5]*3
        result.dist.length *= mul
        result.dist.num_grid = int(mul * result.dist.num_grid)
        result.ave_ele = new_diele(diele_dist.ave_ele, mul)
        ave_static = new_diele(diele_dist.ave_static, mul)
        result.ave_ion = list(np.array(ave_static) - np.array(result.ave_ele))

        result.dist.step_left += vac_len / 2.
        result.dist.step_right += vac_len / 2.

        result.to_json_file(suffix=str(round(mul, 1)))


def new_diele(old_diele: List[float], mul: float):
    return [new_pal_diele(old_diele[0], mul),
            new_pal_diele(old_diele[1], mul),
            new_perp_diele(old_diele[2], mul)]


def new_pal_diele(old_pal_diele: float, mul: float):
    return (old_pal_diele + (mul - 1)) / mul


def new_perp_diele(old_perp_diele: float, mul: float):
    return mul * old_perp_diele / (1 + (mul - 1) * old_perp_diele)


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--diele_const_dist", required=True, type=loadfn,
        help="diele_const_dist.json file.")
    parser.add_argument(
        "-m", "--muls", required=True, type=float, nargs="+",
        help="Multiplicities")
    parser.set_defaults(func=extend_diele_dist_main)
    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


