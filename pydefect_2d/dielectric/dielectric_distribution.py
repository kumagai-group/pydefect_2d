# -*- coding: utf-8 -*- #  Copyright (c) 2023 Kumagai group.
from dataclasses import dataclass
from functools import cached_property
from typing import List

from monty.json import MSONable
from scipy.fftpack import fft
from tabulate import tabulate
from vise.util.mix_in import ToJsonFileMixIn

from pydefect_2d.dielectric.distribution import Dist


@dataclass
class DielectricConstDist(MSONable, ToJsonFileMixIn):
    """Dielectric constant as a function of the z position

    ave_ele: Calculated electronic dielectric constants using a slab model.
    ave_ion: Calculated ionic dielectric constants using a slab model.

    The z-direction is normal to the surfaces.
    """
    ave_ele: List[float]  # [x, y, z] include vacuum permittivity.
    ave_ion: List[float]
    dist: Dist

    @property
    def grid_points(self):
        return self.dist.grid_points()

    @property
    def ave_static(self) -> List[float]:
        return [self.ave_static_x, self.ave_static_y, self.ave_static_z]

    @property
    def ave_static_x(self) -> float:
        return self.ave_ele[0] + self.ave_ion[0]

    @property
    def ave_static_y(self) -> float:
        return self.ave_ele[1] + self.ave_ion[1]

    @property
    def ave_static_z(self) -> float:
        return self.ave_ele[2] + self.ave_ion[2]

    @cached_property
    def static(self):
        return [self.dist.diele_in_plane_scale(self.ave_static_x),
                self.dist.diele_in_plane_scale(self.ave_static_y),
                self.dist.diele_out_of_plane_scale(self.ave_static_z)]

    @cached_property
    def electronic(self):
        return [self.dist.diele_in_plane_scale(self.ave_ele[0]),
                self.dist.diele_in_plane_scale(self.ave_ele[1]),
                self.dist.diele_out_of_plane_scale(self.ave_ele[2])]

    @cached_property
    def effective(self):
        result = []
        for s, e in zip(self.static, self.electronic):
            inner = []
            for ss, ee in zip(s, e):
                aa = 1/ee - 1/ss
                if abs(aa) < 1e-7:
                    inner.append(10**7)
                else:
                    inner.append(1/aa)
            result.append(inner)
        return result

    @cached_property
    def reciprocal_static(self):
        return [fft(e) for e in self.static]

    @cached_property
    def reciprocal_effective(self):
        return [fft(e) for e in self.effective]

    @cached_property
    def reciprocal_static_z(self):
        return fft(self.static[2])

    @cached_property
    def reciprocal_effective_z(self):
        return fft(self.effective[2])

    def __str__(self):
        result = []
        header = ["pos (Å)"]
        for e in ["ε_0"]:
            for direction in ["x", "y", "z"]:
                header.append(f"{e}_{direction}")
        list_ = []
        for i, pos in enumerate(self.grid_points):
            list_.append([pos])
            for e in [self.static]:
                list_[-1].extend([e[0][i], e[1][i], e[2][i]])

        result.append(tabulate(list_, tablefmt="plain", floatfmt=".2f",
                               headers=header))

        return "\n".join(result)

    def to_plot(self, ax):
        ax.set_ylabel("$\epsilon$ ($\epsilon_{vac}$)")
        for e, direction in zip(self.static, ["x", "y", "z"]):
            ax.plot(self.grid_points, e, label=f"ε_0_{direction}")
        ax.legend()

