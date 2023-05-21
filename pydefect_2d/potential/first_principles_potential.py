# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from typing import List

from monty.json import MSONable
from scipy.ndimage import uniform_filter1d
from vise.util.mix_in import ToJsonFileMixIn


@dataclass
class FirstPrinciplesPotentialProfile(MSONable, ToJsonFileMixIn):
    """
    Assume that the vacuum and slab centers locate at z=0 and z=0.5
    """
    z_length: float
    defect_position_in_frac_coord: float
    z_grid: List[float]
    xy_ave_potential: List[float]
    num_grid_per_unit: int

    @property
    def macroscopic_average(self):
        # Note that when num_grid_per_unit is even number,
        # the backward value is used.
        return uniform_filter1d(self.xy_ave_potential,
                                size=self.num_grid_per_unit,
                                mode="wrap").tolist()

    @property
    def macro_pot_at_farthest_point(self):
        pass


