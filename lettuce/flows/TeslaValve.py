"""
TeslaValve Flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary
from lettuce.boundary import EquilibriumBoundaryPU
from lettuce.boundary import AntiBounceBackOutlet
from PIL import Image

class TeslaValve2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice, initialize_with_zeros=True):
        self.resolution = resolution
        self.lattice = lattice
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.initialize_with_zeros = initialize_with_zeros
        self.valve = []

    def initial_solution(self, grid):
        p = np.array([0 * grid[0]], dtype=float)
        u = np.array([0 * grid[0], 0 * grid[1]], dtype=float)
        return p, u

    @property
    def grid(self):

        image = Image.open("/home/mbedru3m/Downloads/TeslaValve_v2.png")
        self.valve = np.asarray(image)[::2, ::2, 0]
        # x = np.linspace(0, 1, num=self.resolution+1, endpoint=True)
        # y = np.linspace(0, 1, num=self.resolution+1, endpoint=True)
        x = np.linspace(0, 1, num=self.valve.shape[0], endpoint=True)
        y = np.linspace(0, 1, num=self.valve.shape[1], endpoint=True)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        mask = np.zeros(self.valve.shape, dtype=bool)
        mask = np.where(np.flip(self.valve,0) < 250, True, False)
        # mask = np.where(self.valve < 250, True, False)
        boundary = BounceBackBoundary(mask=mask, lattice=self.units.lattice)
        mask = np.zeros(self.valve.shape, dtype=bool)
        mask[5,:] = True
        # mask[33,58:65] =True
        boundary_eq = EquilibriumBoundaryPU(mask=mask, lattice=self.units.lattice, units=self.units, velocity=[0.05,0], pressure=0)
        boundary_auslass = AntiBounceBackOutlet(lattice=self.units.lattice, direction=[1,0])
        return [boundary, boundary_eq, boundary_auslass]

    @property
    def acceleration(self):
        return np.array([0.01, 0])

