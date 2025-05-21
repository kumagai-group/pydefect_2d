# -*- coding: utf-8 -*-
#  Copyright (c) 2024 Kumagai group.
from typing import Tuple

import numpy as np
from pymatgen.io.vasp import Chgcar


def count_chg_density(chgcar: Chgcar, range_: Tuple[float, float]) -> float:
    """
    Integrate the planar charge density within a fractional-z range and
    normalise by the total number of grid planes (len(z)), reproducing the
    behaviour of the original script.

    Parameters
    ----------
    chgcar : Chgcar
        Pymatgen Chgcar object containing charge density.
    range_  : Tuple[float, float]
        Fractional z-coordinates that delimit the desired slab.
        If range_[0] > range_[1], the range is assumed to wrap around the
        periodic boundary (e.g. 0.8 → 1.0 and 0.0 → 0.2).

    Returns
    -------
    float
        Integrated charge in the specified range divided by the total number
        of z grid points (len(z)).
    """
    z_density = chgcar.get_average_along_axis(ind=2)
    n_grid = len(z_density)

    z_grid = np.linspace(0.0, 1.0, n_grid, endpoint=False)

    idx_start = int(np.argmin(np.abs(z_grid - range_[0])))
    idx_end = int(np.argmin(np.abs(z_grid - range_[1])))

    if idx_start == idx_end:
        raise ValueError("The specified range is too narrow.")

    if idx_start < idx_end:
        charge = float(z_density[idx_start:idx_end].sum())
    else:
        charge = float(z_density[idx_start:].sum() + z_density[:idx_end].sum())

    return charge / n_grid
