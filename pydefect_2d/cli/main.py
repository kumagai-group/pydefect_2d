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
    make_step_diele_dist, make_1d_gauss_models, make_gauss_model_from_z, \
    make_gaussian_energies_from_args, make_1d_fp_potential, make_1d_slab_model


def add_2d_sub_parser(_argparse, name: str):
    result = _argparse.ArgumentParser(description="", add_help=False)
    if name == "denominator":
        result.add_argument(
            "--denominator", type=int, default=1,
            help="Denominator of FFT grid.")
    elif name == "std_dev":
        result.add_argument(
            "-s", "--std_dev", type=float, required=True,
            help="Standard deviation of the (integrand) gaussian smearing [Å].")
    elif name == "diele_dist":
        result.add_argument(
            "-dd", "--diele_dist", type=loadfn, required=True,
            help="dielectric_const_dist.json file")
    elif name == "gauss_charge_std_dev":
        result.add_argument(
            "--std_dev", default=0.5, type=float,
            help="Standard deviation of the gaussian charge [Å].")
    elif name == "perfect_slab":
        result.add_argument(
            "-p", "--perfect_slab", type=Structure.from_file,
            help="POSCAR file of the perfect slab model.")
    elif name == "perfect_locpot":
        result.add_argument(
            "-pl", "--perfect_locpot", type=Locpot.from_file,
            help="LOCPOT file from a perfect supercell calculation.")
    elif name == "num_grid":
        result.add_argument(
            "-n", "--num_grid", type=int,
            help="Number of FFT grid.")
    elif name == "center":
        result.add_argument(
            "-c", "--center", type=float, required=True,
            help="Center position of layer in fractional coordinates.")
    elif name == "isolated_gauss":
        result.add_argument(
            "--k_max", type=float, default=5.0,
            help="Max of k integration range.")
        result.add_argument(
            "--k_mesh_dist", type=float, default=0.05,
            help="k mesh distance.")
    elif name == "corr_dir":
        result.add_argument(
            "-cd", "--correction_dir", type=Path,
            default=Path.cwd(), help="correction directory.")
    elif name == "one_d_dir":
        result.add_argument(
            "-od", "--one_d_dir", required=True, type=Path,
            help="1d_gauss directory path.")
    elif name == "no_multiprocess":
        result.add_argument(
            "--no_multiprocess", dest="multiprocess", action="store_false",
            help="Switch of the multiprocess.")
    else:
        raise ValueError
    return result


def parse_args_main_vasp(args):

    parser = argparse.ArgumentParser(epilog="",
                                     description="""       
    The command is used for analyzing VASP output files for pydefect_2d.""")

    subparsers = parser.add_subparsers()

    unitcell_parser = add_sub_parser(argparse, name="unitcell")
    supercell_info_parser = add_sub_parser(argparse, name="supercell_info")
    dir_parser = add_sub_parser(argparse, name="dirs")

    dielectric_dist = add_2d_sub_parser(argparse, "diele_dist")
    gauss_charge_std_dev = add_2d_sub_parser(argparse, "gauss_charge_std_dev")
    perfect_locpot = add_2d_sub_parser(argparse, "perfect_locpot")
    center = add_2d_sub_parser(argparse, "center")
    denominator = add_2d_sub_parser(argparse, "denominator")
    std_dev = add_2d_sub_parser(argparse, "std_dev")
    corr_dir = add_2d_sub_parser(argparse, "corr_dir")
    isolated_gauss = add_2d_sub_parser(argparse, "isolated_gauss")
    no_multiprocess = add_2d_sub_parser(argparse, "no_multiprocess")
    one_d_dir = add_2d_sub_parser(argparse, "one_d_dir")

    # --------------------------------------------------------------------------
    parser_gauss_diele_dist = subparsers.add_parser(
        name="gauss_diele_dist",
        description="Make gauss dielectric distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[unitcell_parser, perfect_locpot, denominator,
                 std_dev],
        aliases=['gdd'])
    parser_gauss_diele_dist.add_argument(
        "-c", "--center", type=float, required=True, nargs="+",
        help="Center positions of layer in fractional coordinates.")
    parser_gauss_diele_dist.set_defaults(
        func=make_gauss_diele_dist)

    # --------------------------------------------------------------------------
    parser_make_step_dielectric_dist = subparsers.add_parser(
        name="step_diele_dist",
        description="Make step-like dielectric distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[unitcell_parser, center, perfect_locpot, denominator,
                 std_dev],
        aliases=['sdd'])

    parser_make_step_dielectric_dist.add_argument(
        "-w", "--step_width", type=float, required=True,
        help="Width of step function [Å]")
    parser_make_step_dielectric_dist.add_argument(
        "-wz", "--step_width_z", type=float,
        help="Width of step function for epsilon_z [Å]")
    parser_make_step_dielectric_dist.set_defaults(
        func=make_step_diele_dist)

    # --------------------------------------------------------------------------
    parser_1d_gauss_models = subparsers.add_parser(
        name="1d_gauss_models",
        description=f"Make 1D Gauss models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dielectric_dist, gauss_charge_std_dev, supercell_info_parser],
        aliases=['1gm'])

    parser_1d_gauss_models.add_argument(
        "-m", "--mesh_distance", type=float, default=0.01,
        help="Mesh distance between charge positions in fractional coord.")
    parser_1d_gauss_models.add_argument(
        "-r", "--range", type=float, nargs=2, required=True,
        help="Position range of gauss charge in fractional coord.")
    parser_1d_gauss_models.set_defaults(func=make_1d_gauss_models)

    # --------------------------------------------------------------------------
    parser_gauss_model = subparsers.add_parser(
        name="gauss_model_from_z",
        description=f"Make Gauss models at given z sites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[supercell_info_parser, dielectric_dist, gauss_charge_std_dev,
                 isolated_gauss, corr_dir, no_multiprocess],
        aliases=['gmz'])

    parser_gauss_model.add_argument(
        "-z", "--z_pos", type=float, nargs="+", required=True,
        help="Positions gauss models along z in frac coords.")
    parser_gauss_model.set_defaults(func=make_gauss_model_from_z)

    # --------------------------------------------------------------------------
    parser_gauss_energies = subparsers.add_parser(
        name="gauss_energies",
        description=f"Make Gaussian energies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[corr_dir],
        aliases=['ge'])

    parser_gauss_energies.add_argument(
        "-z", "--z_range", type=float, nargs=2, help="Z range")
    parser_gauss_energies.set_defaults(func=make_gaussian_energies_from_args)

    # --------------------------------------------------------------------------
    parser_make_1d_fp_potential = subparsers.add_parser(
        name="1d_fp_potential",
        description="Make planar averaged potential of first-principles "
                    "calculation result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dir_parser, perfect_locpot, one_d_dir],
        aliases=['1fp'])

    parser_make_1d_fp_potential.set_defaults(func=make_1d_fp_potential)

    # --------------------------------------------------------------------------
    parser_make_1d_slab_model = subparsers.add_parser(
        name="1d_slab_model",
        description="Make planar averaged potential of first-principles "
                    "calculation result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[dir_parser, dielectric_dist, one_d_dir],
        aliases=['1sm'])

    parser_make_1d_slab_model.add_argument(
        "-g", "--gauss_energies", type=loadfn, required=True,
        help="GaussEnergies.json file path.")
    parser_make_1d_slab_model.add_argument(
        "-s", "--slab_center", type=float,
        help="Slab center position in fractional coord. This is used for "
             "estimating the eigenvalue shift.")
    parser_make_1d_slab_model.set_defaults(func=make_1d_slab_model)

    # --------------------------------------------------------------------------
    return parser.parse_args(args)


def main():
    args = parse_args_main_vasp(sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()


