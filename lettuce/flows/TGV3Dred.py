"""
Taylor-Green vortex in 2D and 3D.
"""

import numpy as np

from lettuce.unit import UnitConversion

from lettuce.boundary_TGV import newsuperTGV3D

class SuperReducedTaylorGreenVortex3D:
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution / (1/2*np.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    def initial_solution(self, x):

        u = np.array([
            np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2]),
            -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2]),
            np.zeros_like(np.sin(x[0]))
        ])
        p = np.array([1 / 16. * (np.cos(2 * x[0]) + np.cos(2 * x[1])) * (np.cos(2 * x[2]) + 2)])
        return p, u

    @property
    def grid(self):
        x,dx = np.linspace(0, np.pi/2, num=self.resolution, endpoint=False, retstep=True)
        x=x+dx/2
        y,dy = np.linspace(0, np.pi/2, num=self.resolution, endpoint=False, retstep=True)
        y=y+dy/2
        z,dz = np.linspace(np.pi/2, np.pi, num=self.resolution, endpoint=False, retstep=True)
        z=z+dz/2
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        boundary=newsuperTGV3D(lattice=self.units.lattice)
        return [boundary]