"""
To be discussed
"""

import numpy as np
from lettuce.unit import UnitConversion


class BasicFlow:
    def __init__(self, resolution, reynolds_number, mach_number, lattice, characteristic_length_lu=None):
        self.resolution = resolution
        char_length_lu = resolution if characteristic_length_lu is None else characteristic_length_lu
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu= (2 * np.pi),
            characteristic_velocity_pu=1
        )

    def initial_solution(self, *args, **kwargs):
        u = np.zeros([self.units.lattice.D]+[self.resolution]*self.units.lattice.D)
        p = np.zeros([1]+[self.resolution]*self.units.lattice.D)
        return p, u

    @property
    def grid(self):
        grid = [np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False) for _ in range(3)]
        return np.meshgrid(*grid, indexing='ij')

    @property
    def boundaries(self):
        return []
