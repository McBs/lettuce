"""
Couette Flow
"""

import numpy as np

from lettuce.unit import UnitConversion
from lettuce.boundary import BounceBackBoundary, EquilibriumBoundaryPU
from lettuce.grid import RegularGrid


class CouetteFlow2D(object):
    def __init__(self, resolution, reynolds_number, mach_number, lattice, mpiObject=None):
        self.resolution = resolution
        self.mpiObject=mpiObject
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.rgrid = RegularGrid([resolution, resolution, resolution], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False,mpiObject=mpiObject)


    def refinment(self,newResolution):
        self.resolution=newResolution
        reynolds_number=self.units.reynolds_number
        mach_number=self.units.mach_number
        resolution=newResolution
        lattice=self.units.lattice
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

        self.rgrid = RegularGrid([resolution, resolution, resolution], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False,mpiObject=self.mpiObject)

    def analytic_solution(self, x, t=0):
        # TODO
        raise NotImplementedError

    def initial_solution(self, x):
        return np.array([0*x[0]], dtype=float), np.array([0*x[0],0*x[1]], dtype=float)

    @property
    def grid(self):
        return self.rgrid()

    @property
    def boundaries(self):
        x, y = self.rgrid.global_grid()
        return [EquilibriumBoundaryPU(np.abs(y-1) < 1e-6, self.units.lattice, self.units, np.array([1.0, 0.0])),
                BounceBackBoundary(np.abs(y) < 1e-6, self.units.lattice)]



