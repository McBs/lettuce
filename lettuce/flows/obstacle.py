import numpy as np
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet


class Obstacle2D(object):
    """Flow class to simulate the flow around an object (mask) in 2D. It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    add object mask directly or via "initialize_object" as bool tensor / bool array with true entries forming the object
    char_length_lu: length of the object in flow direction (positive x)"""
    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initialize_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([np.zeros_like(x[0])], dtype=float), np.array(
            [np.where(self.mask == 1, 0, self.units.characteristic_velocity_pu), np.zeros_like(x[1])], dtype=float)

    @property
    def grid(self):
        x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0])),
                AntiBounceBackOutlet(self.units.lattice, [1, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]


class Obstacle3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D. It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    add object mask directly or via "initialize_object" as bool tensor / bool array with true entries forming the object
    char_length_lu: length of the object in flow direction (positive x)"""

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self.mask = None

    def initialise_object(self, mask, lattice):
        self.boundaries.append(BounceBackBoundary(mask, lattice))

    def initial_solution(self, x):
        return np.array([np.zeros_like(x[0])], dtype=float), np.array(
            [np.where(self.mask == 1, 0, self.units.characteristic_velocity_pu), np.zeros_like(x[1]),
             np.where(self.mask == 1, 0, x[2] * 0.1)], dtype=float)

    @property
    def grid(self):
        x = np.linspace(0, self.resolution_x / self.units.characteristic_length_lu, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, self.resolution_y / self.units.characteristic_length_lu, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, self.resolution_z / self.units.characteristic_length_lu, num=self.resolution_z, endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid
        return [EquilibriumBoundaryPU(np.abs(x) < 1e-6, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0, 0])),
                AntiBounceBackOutlet(self.units.lattice, [1, 0, 0]),
                BounceBackBoundary(self.mask, self.units.lattice)]