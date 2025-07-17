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


__all__ = ['TaylorGreenVortex', 'TaylorGreenVortex2D', 'TaylorGreenVortex3D']


class TaylorGreenVortex(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_fneq: bool = True,
                 disrtributed: Optional['disrtributed'] = None):
        self.initialize_fneq = initialize_fneq
        self.disrtributed = disrtributed 
        if self.dist == "mpi":
            print("-----------------Resolution: " + str(resolution) + "------------------")
            # Split the linspace 
            self.split_size = resolution // dist.get_world_size()
            self.remainder = resolution % dist.get_world_size()

            self.upperfill_big = 8
            self.lowerfill_big = 8
            self.upperfill_small = 8
            self.lowerfill_small = 8
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
        if self.disrtributed == "mpi":
            print("Multi node function")
            print(self.stencil.d)
            print(self.resolution)
            endpoints = [2 * torch.pi * (1 - 1 / n ) for n in
                        self.resolution] 
            #create linspace for x-axis
            x_axis = torch.linspace(0, endpoints[0],
                                        steps=self.resolution[0],
                                        device=self.context.device,
                                        dtype=self.context.dtype)

            if self.split_size < 16:
                warnings.warn("Chunk Size too small,"
                              "size must be at least 16", UserWarning)
            if self.remainder > 0:
                

                bigsplits = [x_axis[i*(self.split_size + 1) : (i+1)*(self.split_size + 1)] for i in range(self.remainder)]

                smallsplits = [x_axis[(i*self.split_size + remainder) : ((i+1)*self.split_size + self.remainder)] for i in range(self.remainder, dist.get_world_size())]
            
                self.upperfill_big = int((16 - (self.split_size + 1) % 16)/2)
                self.lowerfill_big = 8 - int(((self.split_size + 1) % 16)/2)
                self.upperfill_small = int((16 - self.split_size % 16)/2)
                self.lowerfill_small = 8 - int((self.split_size % 16)/2)
                
                print("Fills")
                print("uppsefill_big = " + str(self.upperfill_big))
                print("lowerfill_big = " + str(self.lowerfill_big))
                print("upperfill_small = " + str(self.upperfill_small))
                print("lowerfill_small = " + str(self.lowerfill_small))
        
                extended_splits = []

                print("range(remainder):" + str(range(self.remainder)))
                for i in range(self.remainder):
                    left_neighbor = bigsplits[i-1][-self.lowerfill_big:] if i > 0 else smallsplits[-1][-self.lowerfill_big:]  # Get last value of previous (or last split for first one)
                    right_neighbor = bigsplits[i+1][:self.upperfill_big] if i < self.remainder - 1 else smallsplits[0][:self.upperfill_big]  # Get first value of next (or first split for last one)

                    extended_split = torch.cat([left_neighbor, bigsplits[i], right_neighbor])
                    extended_splits.append(extended_split)

                print("len(range(remainder, dist.get_world_size())):" + str(len(range(self.remainder, dist.get_world_size()))))
                for i in range(len(range(self.remainder, dist.get_world_size()))):
                    left_neighbor = smallsplits[i-1][-self.lowerfill_small:] if i > 0 else bigsplits[-1][-self.lowerfill_small:]  # Get last value of previous (or last split for first one)
                    right_neighbor = smallsplits[i+1][:self.upperfill_small] if i < len(range(self.remainder, dist.get_world_size())) - 1 else bigsplits[0][:self.upperfill_small]  # Get first value of next (or first split for last one)

                    extended_split = torch.cat([left_neighbor, smallsplits[i], right_neighbor])
                    extended_splits.append(extended_split)


            else:
                splits = [x_axis[i*self.split_size : (i+1)*self.split_size] for i in range(dist.get_world_size())]
                filename = "/home/mbecke3g/data/split" + str(dist.get_rank()) + ".pt"
                torch.save(splits, filename)
                extended_splits = []
                for i in range(dist.get_world_size()):
                    left_neighbor = splits[i-1][-8:] if i > 0 else splits[-1][-8:]  # Get last value of previous (or last split for first one)
                    right_neighbor = splits[i+1][:8] if i < dist.get_world_size() - 1 else splits[0][:8]  # Get first value of next (or first split for last one)

                    extended_split = torch.cat([left_neighbor, splits[i], right_neighbor])
                    extended_splits.append(extended_split)
                    
            
            yz =  tuple(torch.linspace(0, endpoints[n],
                                steps=self.resolution[n],
                                device=self.context.device,
                                dtype=self.context.dtype)        
                        for n in range(self.stencil.d - 1))

            xyz = (extended_splits[dist.get_rank()],) + yz

            filename = "/home/mbecke3g/data/meshgrid" + str(dist.get_rank()) + ".pt"
            torch.save(torch.meshgrid(*xyz, indexing='ij'), filename)
            return torch.meshgrid(*xyz, indexing='ij')    
        else:
            print("singel node function")
            endpoints = [2 * torch.pi * (1 - 1 / n) for n in
                        self.resolution]  # like endpoint=False in np.linspace
            xyz = tuple(torch.linspace(0, endpoints[n],
                                    steps=self.resolution[n],
                                    device=self.context.device,
                                    dtype=self.context.dtype)
                        for n in range(self.stencil.d))
            filename = "/home/mbecke3g/data/xyz_serial.pt"
            torch.save(xyz, filename)
            print("------xyz (single)-----")
            print(xyz)
            filename = "/home/mbecke3g/data/meshgrid_serial.pt"
            torch.save(torch.meshgrid(*xyz, indexing='ij'), filename)
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


def TaylorGreenVortex3D(context: 'Context', resolution: Union[int, List[int]],
                        reynolds_number, mach_number,
                        stencil: Optional['Stencil'] = None,
                        equilibrium: Optional['Equilibrium'] = None):
    warnings.warn("TaylorGreenVortex3D is deprecated. Use TaylorGreenVortex"
                  " instead", DeprecationWarning)
    return TaylorGreenVortex(context=context, resolution=resolution,
                             reynolds_number=reynolds_number,
                             mach_number=mach_number, stencil=stencil,
                             equilibrium=equilibrium)


def TaylorGreenVortex2D(context: 'Context', resolution: Union[int, List[int]],
                        reynolds_number, mach_number,
                        stencil: Optional['Stencil'] = None,
                        equilibrium: Optional['Equilibrium'] = None):
    warnings.warn("TaylorGreenVortex2D is deprecated. Use TaylorGreenVortex"
                  " instead", DeprecationWarning)
    return TaylorGreenVortex(context=context, resolution=resolution,
                             reynolds_number=reynolds_number,
                             mach_number=mach_number, stencil=stencil,
                             equilibrium=equilibrium)
