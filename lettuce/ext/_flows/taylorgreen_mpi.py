"""
Taylor-Green vortex in 2D and 3D.
"""

import warnings
from typing import Union, List, Optional

import torch
import torch.distributed as dist

from ... import UnitConversion
from .._stencil import D2Q9
from . import ExtFlow


__all__ = ['TaylorGreenVortex_mpi', 'TaylorGreenVortex2D_mpi', 'TaylorGreenVortex3D_mpi']


class TaylorGreenVortex_mpi(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_fneq: bool = True):
        self.initialize_fneq = initialize_fneq
        if stencil is None and not isinstance(resolution, list):
            warnings.warn("Requiring information about dimensionality!"
                          " Either via stencil or resolution. Setting "
                          "dimension to 2.", UserWarning)
            self.stencil = D2Q9()
        else:
            self.stencil = stencil() if callable(stencil) else stencil
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * self.stencil.d
        else:
            assert len(resolution) in [2, 3], ('the resolution of a '
                                               'taylor-green-vortex '
                                               'must be 2- or 3-dimensional!')
            return resolution

    def make_units(self, reynolds_number, mach_number,
                   resolution) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=2 * torch.pi,
            characteristic_velocity_pu=1)

    @property
    def grid(self):
        print("--------Worldsize----------")
        print(dist.get_world_size())
        print("--------Rank----------")
        print(dist.get_rank())
        print("--------resulution----------")
        print(self.resolution[0])
        nodes = int(self.resolution[0] / dist.get_world_size())
        print("--------nodes----------")
        print(nodes)
        endpoints = [2 * torch.pi * (1 - 1 / n) for n in
                     self.resolution]  # like endpoint=False in np.linspace
        print("--------endpoints----------")
        print(endpoints)
        print("--------self.stencil.d-------")
        print(self.stencil.d)
        lspace=torch.split(torch.linspace(0, endpoints[0],
                                       steps=self.resolution[0],
                                       device=self.context.device,
                                       dtype=self.context.dtype), nodes)
        
        for n in range(1, self.stencil.d): 
            lspace=torch.linspace(0, endpoints[n],
                                       steps=self.resolution[n],
                                       device=self.context.device,
                                       dtype=self.context.dtype)
        print("--------lspace----------")
        print(lspace)

        xyz_test = tuple(lspace)
        print("--------xyz with split lspace----------")
        print(xyz_test)


        xyz = tuple(torch.linspace(0, endpoints[n],
                                   steps=self.resolution[n],
                                   device=self.context.device,
                                   dtype=self.context.dtype)
                    for n in range(self.stencil.d))
        print("--------xyz----------")
        print(xyz)
        return torch.meshgrid(*xyz, indexing='ij')

    def initial_pu(self) -> (torch.Tensor, torch.Tensor):
        return self.analytic_solution(t=0)

    def analytic_solution(self, t: float) -> (torch.Tensor, torch.Tensor):
        if t > 0 and self.stencil.d > 2:
            warnings.warn("The analytic solution is only true for the 2D TGV!")
        grid = self.grid
        nu = self.context.convert_to_tensor(self.units.viscosity_pu)
        if len(self.resolution) == 2:
            u = torch.stack(
                [torch.cos(grid[0])
                 * torch.sin(grid[1])
                 * torch.exp(-2 * nu * t),
                 -torch.sin(grid[0])
                 * torch.cos(grid[1])
                 * torch.exp(-2 * nu * t)])
            p = -torch.stack(
                [0.25 * (torch.cos(2 * grid[0]) + torch.cos(2 * grid[1]))
                 * torch.exp(-4 * nu * t)])
        else:
            u = torch.stack(
                [torch.sin(grid[0])
                 * torch.cos(grid[1])
                 * torch.cos(grid[2]),
                 -torch.cos(grid[0])
                 * torch.sin(grid[1])
                 * torch.cos(grid[2]),
                 torch.zeros_like(grid[0])])
            p = torch.stack(
                [1 / 16. * (torch.cos(2 * grid[0]) + torch.cos(2 * grid[1]))
                 * (torch.cos(2 * grid[2]) + 2)])
        return p, u

    @property
    def boundaries(self) -> List['Boundary']:
        return []


def TaylorGreenVortex3D_mpi(context: 'Context', resolution: Union[int, List[int]],
                        reynolds_number, mach_number,
                        stencil: Optional['Stencil'] = None,
                        equilibrium: Optional['Equilibrium'] = None):
    warnings.warn("TaylorGreenVortex3D is deprecated. Use TaylorGreenVortex"
                  " instead", DeprecationWarning)
    return TaylorGreenVortex_mpi(context=context, resolution=resolution,
                             reynolds_number=reynolds_number,
                             mach_number=mach_number, stencil=stencil,
                             equilibrium=equilibrium)


def TaylorGreenVortex2D_mpi(context: 'Context', resolution: Union[int, List[int]],
                        reynolds_number, mach_number,
                        stencil: Optional['Stencil'] = None,
                        equilibrium: Optional['Equilibrium'] = None):
    warnings.warn("TaylorGreenVortex2D is deprecated. Use TaylorGreenVortex"
                  " instead", DeprecationWarning)
    return TaylorGreenVortex_mpi(context=context, resolution=resolution,
                             reynolds_number=reynolds_number,
                             mach_number=mach_number, stencil=stencil,
                             equilibrium=equilibrium)
