"""Lattice Boltzmann Solver"""

from lettuce.mpiClass import mpiObject
from timeit import default_timer as timer

from numpy.random import f
from lettuce import (
    LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, lattices, torch_gradient, BounceBackBoundary,
)
import lettuce
from lettuce.util import grid_coarse_to_Fine, pressure_poisson
import pickle
from copy import deepcopy
import warnings
import torch
import numpy as np

__all__ = ["Simulation"]


class Simulation:
    """High-level API for simulations.

    Attributes
    ----------
    reporters : list
        A list of reporters. Their call functions are invoked after every simulation step (and before the first one).

    """

    def __init__(self, flow, lattice, collision, streaming,mpiObject=None,nan_steps=None):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0
        if(mpiObject is not None):
            self.mpiObject=mpiObject
        else:
            self.mpiObject=lettuce.mpiObject(0)
        if(self.mpiObject.initOnCPU==1):
            
            self.latticeplanned=self.lattice
            lattice=lettuce.Lattice(lattice.stencil, device=torch.device("cpu"), dtype=lattice.dtype)
            self.lattice=lattice

        grid = flow.grid
        print(grid[0].shape)
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")

    
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))
        
        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(grid[0], dtype=bool))
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
        
        
        #make imports for distributed execution and apply boundaries + set wich step-Methode to run
        if(self.mpiObject.mpi==1):
            print("mpisim")
            global os
            import os
            global Process
            from torch.multiprocessing import Process
            global dist
            import torch.distributed as dist

            #set rank and size
            self.rank=self.mpiObject.rank
            self.size=self.mpiObject.size
            self.nan_steps = nan_steps
            self.index = [self.mpiObject.index, ...]
            self.nan_cnt = 0
            print(f"Process {self.rank} covers {self.index}")

            # set which Methode should be executed
            self.runStep=self.stepMPI

        else:
            self.runStep=self.stepNonMPI
        # Apply boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
        removeentrys=[]
        for i in range (len(self._boundaries)):
            boundary=self._boundaries[i]
            if hasattr(boundary, "make_no_collision_mask"):
                bound= boundary.make_no_collision_mask(self.f.shape)
                bound=lattice.convert_to_tensor(bound)
                self.no_collision_mask = self.no_collision_mask | bound
            if hasattr(boundary, "make_no_stream_mask"):
                
                bound= boundary.make_no_stream_mask(torch.Size(self.f.shape))
                bound=lattice.convert_to_tensor(bound)
                no_stream_mask = no_stream_mask | bound
            if(not boundary.hasTrueEntrys()):
                removeentrys.append(i)

        for i in range(len(removeentrys)-1,-1,-1):
            entry=removeentrys[i]
            self._boundaries.pop(entry) 

        
        if no_stream_mask.any():
            self.streaming.no_stream_mask = no_stream_mask
                
            
    
    def step(self,num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        
        #mask abfrage hier

        #Grid refinment
        if(self.mpiObject.gridRefinment==1):
            num_steps=num_steps*2
            print("Now simulating ",num_steps, " steps")

            resolution=self.flow.resolution
            if(self.lattice.D==2):
                grid=lettuce.RegularGrid([resolution,resolution],1,1,mpiObject=self.mpiObject)
            else:
                grid=lettuce.RegularGrid([resolution,resolution,resolution],1,1,mpiObject=self.mpiObject)
            index = self.mpiObject.index
            xDim=0
            if(self.mpiObject.mpi==1 and self.mpiObject.distributefromRank0==1):
                self.rank=0
                self.size=1
                xDim=1
                
                
                #distribute f
               
                self.f=self.flow.rgrid.distributeToList(self.f)
                
                #select my part
                newindex=slice(index.start*2,index.stop*2)
                self.mpiObject.index=newindex
                self.f=self.streaming.transmitWhole5Parts(self.f)

            #grid refinment
            taucorse=self.flow.units.relaxation_parameter_lu
            self.flow.refinment(resolution*2)
            
            taufine=self.flow.units.relaxation_parameter_lu
            if(self.lattice.D==2):
                self.f=grid_coarse_to_Fine(self.lattice,self.f,taucorse,taufine,[2,index.stop-index.start+10,resolution],xDim)
            else:
                self.f=grid_coarse_to_Fine(self.lattice,self.f,self.flow.units.relaxation_parameter_lu,[2,index.stop-index.start+10,resolution,resolution],xDim)
            grid = self.flow.grid
            self.no_collision_mask = self.lattice.convert_to_tensor(np.zeros_like(grid[0], dtype=bool))
            no_stream_mask = self.lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
            
            #boundarys scalieren

            self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
            removeentrys=[]
            for i in range (len(self._boundaries)):
                boundary=self._boundaries[i]
                if hasattr(boundary, "make_no_collision_mask"):
                    self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
                if hasattr(boundary, "make_no_stream_mask"):
                    no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
                if(not boundary.hasTrueEntrys()):
                   removeentrys.append(i)
            if no_stream_mask.any():
                self.streaming.no_stream_mask = no_stream_mask

            for i in range(len(removeentrys)-1,-1,-1):
                entry=removeentrys[i]
                self._boundaries.pop(entry) 
        else:
            #check for distributing
            if(self.mpiObject.mpi==1 and self.mpiObject.distributefromRank0==1):
                self.f=self.flow.rgrid.distributeToList(self.f)
                resolution=self.flow.resolution
                self.flow.refinment(resolution)
                grid = self.flow.grid
                self.no_collision_mask = self.lattice.convert_to_tensor(np.zeros_like(grid[0], dtype=bool))
                no_stream_mask = self.lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
                
                #boundarys scalieren

                self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state
                removeentrys=[]
                for i in range (len(self._boundaries)):
                    boundary=self._boundaries[i]
                    if hasattr(boundary, "make_no_collision_mask"):
                        self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
                    if hasattr(boundary, "make_no_stream_mask"):
                        no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
                    if(not boundary.hasTrueEntrys()):
                        removeentrys.append(i)
                if no_stream_mask.any():
                    self.streaming.no_stream_mask = no_stream_mask

                for i in range(len(removeentrys)-1,-1,-1):
                    entry=removeentrys[i]
                    self._boundaries.pop(entry) 
        
        #copy self.f and masks to target device
        if(self.mpiObject.initOnCPU==1):
            self.lattice=self.latticeplanned 
            del(self.latticeplanned)
            self.f=self.lattice.convert_to_tensor(self.f)
            
            self.no_collision_mask =self.lattice.convert_to_tensor(self.no_collision_mask)  
            no_stream_mask = self.streaming.no_stream_mask
            if(no_stream_mask is not None):
                if no_stream_mask.any():
                    no_stream_mask=self.lattice.convert_to_tensor(no_stream_mask)  
                    self.streaming.no_stream_mask = no_stream_mask
        #execute
        return self.runStep(num_steps)


    def stepNonMPI(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1
            self.f = self.streaming(self.f)
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            for boundary in self._boundaries:
                self.f = boundary(self.f)
            self._report()
        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def stepMPI(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()
        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1   
            self.f = self.streaming(self.f)
            self.f =self.streaming.stream(self.f)
            self.f=self.streaming.reduce(self.f)
            self._report()
            
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            
            for boundary in self._boundaries:
                self.f = boundary(self.f)
            

            if self.nan_steps is not None:
                if torch.isnan(self.f).any():
                    print("nan detected")
                    for reporter in self.reporters:
                        if hasattr(reporter, "nan_out"):
                            reporter.interval = 10
                    self.nan_cnt += 1
                    if self.nan_cnt > self.nan_steps:
                        print("Simulation is being cancelled because an f-value is nan.")
                        return None
        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)

    def initialize(self, max_num_steps=500, tol_pressure=0.001):
        """Iterative initialization to get moments consistent with the initial velocity.

        Using the initialization does not better TGV convergence. Maybe use a better scheme?
        """
        warnings.warn("Iterative initialization does not work well and solutions may diverge. Use with care. "
                      "Use initialize_f_neq instead.",
                      ExperimentalWarning)
        transform = get_default_moment_transform(self.lattice)
        collision = BGKInitialization(self.lattice, self.flow, transform)
        streaming = self.streaming
        p_old = 0
        for i in range(max_num_steps):
            self.f = streaming(self.f)
            self.f = collision(self.f)
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(self.f))
            if (torch.max(torch.abs(p - p_old))) < tol_pressure:
                break
            p_old = deepcopy(p)
        return i

    def initialize_pressure(self, max_num_steps=100000, tol_pressure=1e-6):
        """Reinitialize equilibrium distributions with pressure obtained by a Jacobi solver.
        Note that this method has to be called before initialize_f_neq.
        """
        u = self.lattice.u(self.f)
        rho = pressure_poisson(
            self.flow.units,
            self.lattice.u(self.f),
            self.lattice.rho(self.f),
            tol_abs=tol_pressure,
            max_num_steps=max_num_steps
        )
        self.f = self.lattice.equilibrium(rho, u)

    def initialize_f_neq(self):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
       
        if(self.mpiObject.mpi==0):

            rho = self.lattice.rho(self.f)

            u = self.lattice.u(self.f)
           
        else:
            #instead of transmitting parts we need to Transmit the whole Domaine
            self.f = self.streaming.transmitWholeParts(self.f)
           
            rho = self.lattice.rho(self.f)
            u = self.lattice.u(self.f)
          
        grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]

        grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
        S = torch.cat([grad_u0, grad_u1])

        if self.lattice.D == 3:
            grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
            S = torch.cat([S, grad_u2])   

        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2

        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)

        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        feq = self.lattice.equilibrium(rho, u)
        self.f = feq + fneq
        if(self.mpiObject.mpi==1):
            #reduce f again
            self.f=self.streaming.reduce3(self.f)

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)
