# -*- coding: utf-8 -*-
#  Copyright (c) 2023 Kumagai group.
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass

import numpy as np
from vise.util.logger import get_logger

from pydefect_2d.dielectric.dielectric_distribution import DielectricConstDist
from pydefect_2d.dielectric.distribution import StepDist, ManualDist

logger = get_logger(__name__)


@dataclass
class ExtendDieleDist(ABC):
    orig_diele_dist: DielectricConstDist

    @property
    @abstractmethod
    def diele_const_dist(self):
        pass

    @property
    def new_ave_ele(self):
        return self._new_ave_diele(self.orig_diele_dist.ave_ele)

    @property
    def new_ave_total(self):
        return self._new_ave_diele(self.orig_diele_dist.ave_static)

    @property
    def new_ave_ion(self):
        return [i-j for i, j in zip(self.new_ave_total, self.new_ave_ele)]

    @abstractmethod
    def _new_ave_diele(self, old_ave_diele):
        pass


@dataclass
class AddVacuum(ExtendDieleDist):
    delta_l: float

    @property
    def diele_const_dist(self):
        return DielectricConstDist(ave_ele=self.new_ave_ele,
                                   ave_ion=self.new_ave_ion,
                                   dist=self._modify_dist)

    @property
    def old_l(self):
        return self.orig_diele_dist.dist.length

    @property
    def new_l(self):
        return self.orig_diele_dist.dist.length + self.delta_l

    def _new_ave_diele(self, old_ave_diele):
        return [(old_ave_diele[0] * self.old_l + self.delta_l) / self.new_l,
                (old_ave_diele[1] * self.old_l + self.delta_l) / self.new_l,
                self.new_l / (self.old_l / old_ave_diele[2] + self.delta_l)]

    @property
    def _modify_dist(self):
        assert isinstance(self.orig_diele_dist.dist, StepDist)
        result = copy(self.orig_diele_dist.dist)
        mul = self.new_l / self.old_l
        result.length += self.delta_l
        result.center += self.delta_l / 2
        result.num_grid = int(self.orig_diele_dist.dist.num_grid * mul)
        return result


@dataclass
class RepeatDieleDist(ExtendDieleDist):
    mul: int

    @property
    def diele_const_dist(self):
        return DielectricConstDist(ave_ele=self.new_ave_ele,
                                   ave_ion=self.new_ave_ion,
                                   dist=self._modify_dist)

    def _new_ave_diele(self, old_ave_diele):
        return old_ave_diele

    @property
    def _modify_dist(self):
        orig = self.orig_diele_dist.dist.to_manual_dist
        return ManualDist(orig.length * self.mul,
                          orig.num_grid * self.mul,
                          np.tile(orig.unscaled_in_plane_dist, self.mul),
                          np.tile(orig.unscaled_out_of_plane_dist, self.mul))
