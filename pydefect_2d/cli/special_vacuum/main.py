# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys

from monty.serialization import loadfn
from pydefect.cli.main import add_sub_parser
from pymatgen.io.vasp import Vasprun

from pydefect_2d.cli.main import add_2d_sub_parser
from pydefect_2d.cli.special_vacuum.main_function import \
    extend_dielectric_const_dist, calc_special_vacuum


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The utility command in pydefect_2d to determine a special vacuum length.""")

    subparsers = parser.add_subparsers()
    dir_parser = add_sub_parser(argparse, name="dirs")
    dielectric_dist = add_2d_sub_parser(argparse, "diele_dist")


# --------------------------------------------------------------------------
    parser_extend_dielectric_dist = subparsers.add_parser(
        name="extend_dielectric_dist",
        description="Make dielectric constant distributions with different"
                    "lengths along the normal direction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist],
        aliases=['edd'])
    parser_extend_dielectric_dist.add_argument(
        "-l", "--slab_lengths", type=float,
        help="Out-of-plane slab lengths in angstrom.")
    parser_extend_dielectric_dist.add_argument(
        "-gcm", "--gauss_charge_model", type=loadfn, required=True,
        help="gauss_charge_model.json file")
    parser_extend_dielectric_dist.set_defaults(
        func=extend_dielectric_const_dist)

    # --------------------------------------------------------------------------
    parser_sv_length = subparsers.add_parser(
        name="special_vacuum",
        description="Calculate the special vacuum length from a series of"
                    "supercell models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dir_parser],
        aliases=["sv"])
    parser_sv_length.add_argument(
        func=calc_special_vacuum)
    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


