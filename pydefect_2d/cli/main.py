# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.

import argparse
import sys
from pathlib import Path

from monty.serialization import loadfn
from pydefect.cli.main import add_sub_parser
from pymatgen.core import Structure
from pymatgen.io.vasp import Locpot

from pydefect_2d.cli.main_function import make_gauss_diele_dist, \
    make_step_diele_dist, make_fp_1d_potential, make_slab_model, \
    make_1d_gauss_models, make_gauss_model


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for analyzing VASP output files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    pcr_parser = add_sub_parser(argparse, name="perfect_calc_results")
    unitcell_parser = add_sub_parser(argparse, name="unitcell")
    dir_parser = add_sub_parser(argparse, name="dirs")
    supercell_info_parser = add_sub_parser(argparse, name="supercell_info")

    gauss_charge_model = argparse.ArgumentParser(description="", add_help=False)
    gauss_charge_model.add_argument(
        "-g", "--gauss_charge_model", required=True, type=loadfn,
        help="gauss_charge_model.json file")

    dielectric_dist = argparse.ArgumentParser(description="", add_help=False)
    dielectric_dist.add_argument(
        "-dd", "--diele_dist", type=loadfn,
        help="dielectric_const_dist.json file")

    gauss_charge_std_dev = \
        argparse.ArgumentParser(description="", add_help=False)
    gauss_charge_std_dev.add_argument(
        "--std_dev", default=0.5, type=float,
        help="Standard deviation of the gaussian charge [Å].")

    perfect_slab = argparse.ArgumentParser(description="", add_help=False)
    perfect_slab.add_argument(
        "-p", "--perfect_slab", required=True, type=Structure.from_file,
        help="POSCAR file of the perfect slab model.")

    perfect_locpot = argparse.ArgumentParser(description="", add_help=False)
    perfect_locpot.add_argument(
        "-pl", "--perfect_locpot", required=True, type=Locpot.from_file,
        help="LOCPOT file from a perfect supercell calculation.")

    mesh_dist = argparse.ArgumentParser(description="", add_help=False)
    mesh_dist.add_argument(
        "-m", "--mesh_distance", type=float, default=0.05,
        help="Mesh distance for the grid along z direction (1/Å).")

    center = argparse.ArgumentParser(description="", add_help=False)
    center.add_argument(
        "-c", "--center", type=float, required=True,
        help="Center position of layer in fractional coordinates.")

    isolated_gauss = argparse.ArgumentParser( description="", add_help=False)
    isolated_gauss.add_argument(
        "--k_max", type=float, default=5.0,
        help="Max of k integration range.")
    isolated_gauss.add_argument(
        "--k_mesh_dist", type=float, default=0.05,
        help="k mesh distance.")

    # --------------------------------------------------------------------------
    parser_gauss_diele_dist = subparsers.add_parser(
        name="gauss_diele_dist",
        description="Make gauss dielectric distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[unitcell_parser, perfect_slab, mesh_dist, center],
        aliases=['gdd'])

    parser_gauss_diele_dist.add_argument(
        "--std_dev", type=float, required=True,
        help="Standard deviation of the gaussian smearing [Å].")
    parser_gauss_diele_dist.set_defaults(
        func=make_gauss_diele_dist)

    # --------------------------------------------------------------------------
    parser_make_step_dielectric_dist = subparsers.add_parser(
        name="step_diele_dist",
        description="Make step-like dielectric distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[unitcell_parser, perfect_slab, mesh_dist, center],
        aliases=['sdd'])

    parser_make_step_dielectric_dist.add_argument(
        "-w", "--step_width", type=float, required=True,
        help="Width of step function [Å]")
    parser_make_step_dielectric_dist.add_argument(
        "--error_func_width", type=float, default=0.3,
        help="Width of error function [Å]")
    parser_make_step_dielectric_dist.set_defaults(
        func=make_step_diele_dist)

    # --------------------------------------------------------------------------
    parser_1d_gauss_models = subparsers.add_parser(
        name="1d_gauss_models",
        description=f"Make 1D Gauss models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, gauss_charge_std_dev, supercell_info_parser,
                 perfect_locpot],
        aliases=['1gm'])

    parser_1d_gauss_models.add_argument(
        "-r", "--range", type=float, nargs=2,
        help="Position range of gauss charge in fractional coord.")
    parser_1d_gauss_models.set_defaults(func=make_1d_gauss_models)

    # --------------------------------------------------------------------------
    parser_fp_1d_potential = subparsers.add_parser(
        name="fp_1d_potential",
        description="Make planar averaged potential of first-principles "
                    "calculation result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dir_parser, perfect_locpot],
        aliases=['fp'])

    parser_fp_1d_potential.add_argument(
        "-p", "--pot_dir", type=Path,
        help="Directory includes gauss1_d_potential.json files.")
    parser_fp_1d_potential.set_defaults(func=make_fp_1d_potential)

    # --------------------------------------------------------------------------
    parser_gauss_model = subparsers.add_parser(
        name="gauss_model",
        description=f"Make Gauss model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, gauss_charge_std_dev, isolated_gauss,
                 dir_parser],
        aliases=['gm'])

    parser_gauss_model.add_argument(
        "--no_multiprocess", dest="multiprocess", action="store_false",
        help="Switch of the multiprocess.")
    parser_gauss_model.set_defaults(func=make_gauss_model)

    # --------------------------------------------------------------------------
    parser_make_slab_model = subparsers.add_parser(
        name="slab_model",
        description="Make slab_model.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, pcr_parser, dir_parser],
        aliases=['sm'])

    parser_make_slab_model.add_argument(
        "-cd", "--correction_dir", required=True, type=Path,
        help="correction director.")
    parser_make_slab_model.set_defaults(func=make_slab_model)

    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


