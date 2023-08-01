# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
import sys

from monty.serialization import loadfn
from pymatgen.core import Structure


def atom_disp(s: Structure, disp: dict) -> Structure:
    result = s.copy()
    for idx, coords in disp.items():
        result.translate_sites(idx, coords, frac_coords=False)

    return result


def disp_from_yaml(yaml_filename):
    return {idx: [float(i) for i in coords.split()]
            for idx, coords in loadfn(yaml_filename).items()}


def main():
    org_poscar = sys.argv[1]
    yaml = sys.argv[2]
    disp = disp_from_yaml(yaml)
    s = atom_disp(Structure.from_file(org_poscar), disp)
    s.to("POSCAR_disp")


if __name__ == "__main__":
    main()
