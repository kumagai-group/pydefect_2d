# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from typing import List

from scipy.ndimage import uniform_filter1d

from pydefect_2d.potential.grids import Grids


@dataclass
class FirstPrinciplesPotentialProfile(Grids):
    """
    Assume that the vacuum and slab centers locate at z=0 and z=0.5
    """
    defect_position_in_frac_coord: float
    xy_ave_potential: List[float]
#    num_grid_per_unit: int

    # @property
    # def macroscopic_average(self):
    #     # Note that when num_grid_per_unit is even number,
    #     # the backward value is used.
    #     return uniform_filter1d(self.xy_ave_potential,
    #                             size=self.num_grid_per_unit,
    #                             mode="wrap").tolist()

    def to_plot(self, ax, add_labels=True):
        ax.plot(self.grids[2], self.xy_ave_potential,
                label="calculated", color="blue")
        if add_labels:
            ax.set_xlabel("Distance (Å)")
            ax.set_ylabel("Potential energy (eV)")

    @property
    def pot_at_farthest_point(self):
        pass


