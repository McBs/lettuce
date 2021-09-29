from lettuce.grid import RegularGrid
from lettuce.streaming import StandardStreaming
import numpy as np
import torch
from torch._C import float32
from lettuce import (
    Lattice, D2Q9, D3Q27, TaylorGreenVortex2D,
    TaylorGreenVortex3D, mpiClass, torch_gradient, grid_fine_to_coarse,
    BGKCollision, Simulation
)

from lettuce import TaylorGreenVortex2D, TaylorGreenVortex3D, PoiseuilleFlow2D, Lattice, D3Q27, D2Q9, write_image, \
    BGKCollision, StandardStreaming, Simulation, DecayingTurbulence
from lettuce.reporters import write_vtk, VTKReporter, \
    ObservableReporter  # ,EnstrophyReporter,EnergyReporter,MaxUReporter,SpectrumReporter
from lettuce.observables import Enstrophy, EnergySpectrum, MaximumVelocity, IncompressibleKineticEnergy, Mass
from lettuce import mpiClass
from lettuce.util import pressure_poisson
import pytest

def test_running():
    #does it run correctly in Single mode?
    torchdevice=torch.device("cpu")
    def checking(device,mpiObject):
        assert (torchdevice==device)
    mpiClass.running(checking,torchdevice)

def test_creatingMPIObj():
    #does it run correctly in Single mode?
    mpi=mpiClass.mpiObject(0)
    assert (mpi.mpi==0)
    

def test_creatingPmpiObj():
    mpi=mpiClass.mpiObject(1)
    assert (mpi.mpi==1)

def test_Sgrid():
    rg=RegularGrid([100,100,100],10,10)

@pytest.mark.mpi
def test_Pgrid():
    def checking(device,mpiObject):
        rg=RegularGrid([100,100,100],10,10,mpiObject=mpiObject)

    mpi=mpiClass.mpiObject(1)
    torchdevice=torch.device("cpu")
    mpiClass.running(checking,torchdevice, mpiObjectInput=mpi)

@pytest.mark.mpi
def test_ParrallelTGV3D():
    torchdevice=torch.device("cpu")
    mpi=mpiClass.mpiObject(1)
    def checking(device,mpiObject):
        
        lattice = Lattice(D2Q9, device=device, dtype=torch.float32)
        flow = TaylorGreenVortex2D(resolution=24, reynolds_number=10, mach_number=0.05, lattice=lattice,mpiObject=mpiObject)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu,mpiObject=mpiObject)
        streaming = StandardStreaming(lattice,mpiObject=mpiObject)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming,mpiObject=mpiObject)
        simulation.step(1)

    mpiClass.running(checking,torchdevice, mpiObjectInput=mpi)
    
@pytest.mark.mpi
def test_WholeStreaming():
    torchdevice=torch.device("cpu")
    mpi=mpiClass.mpiObject(1)
    def checking(device,mpiObject):
        
        lattice = Lattice(D3Q27, device=device, dtype=torch.float32,mpiObject=mpiObject)
        flow = TaylorGreenVortex3D(resolution=24, reynolds_number=10, mach_number=0.05, lattice=lattice,mpiObject=mpiObject)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu,mpiObject=mpiObject)
        streaming = StandardStreaming(lattice,mpiObject=mpiObject)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming,mpiObject=mpiObject)
        f_org=simulation.f
        f=streaming.transmitWholeParts(f_org)
    mpiClass.running(checking,torchdevice, mpiObjectInput=mpi)

@pytest.mark.mpi
def test_Whole5Streaming():
    torchdevice=torch.device("cpu")
    mpi=mpiClass.mpiObject(1)
    def checking(device,mpiObject):
        
        lattice = Lattice(D3Q27, device=device, dtype=torch.float32)
        flow = TaylorGreenVortex3D(resolution=24, reynolds_number=10, mach_number=0.05, lattice=lattice,mpiObject=mpiObject)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
        streaming = StandardStreaming(lattice,mpiObject=mpiObject)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming,mpiObject=mpiObject)
        f_org=simulation.f
        f=streaming.transmitWhole5Parts(f_org)
    mpiClass.running(checking,torchdevice, mpiObjectInput=mpi)

@pytest.mark.mpi
@pytest.mark.parametrize("Observable", [Enstrophy,MaximumVelocity, IncompressibleKineticEnergy, Mass])
@pytest.mark.parametrize("Case", [TaylorGreenVortex2D, TaylorGreenVortex3D])
def test_generic_reporters(Observable, Case):
    torchdevice=torch.device("cpu")
    mpi=mpiClass.mpiObject(1)
    def checking(device,mpiObject):

        lattice = Lattice(D2Q9, dtype=torch.float32, device=device)
        flow = Case(32, 10000, 0.05, lattice=lattice,mpiObject=mpiObject)
        if Case == TaylorGreenVortex3D:
            lattice = Lattice(D3Q27, dtype=torch.float32, device=device,mpiObject=mpiObject)
        collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
        streaming = StandardStreaming(lattice,mpiObject=mpiObject)
        simulation = Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming,mpiObject=mpiObject)
        reporter = ObservableReporter(Observable(lattice, flow), interval=1, out=None,mpiObject=mpiObject)
        simulation.reporters.append(reporter)
        simulation.step(2)
        values = np.asarray(reporter.out)
        if(mpiObject.rank==0):
            assert values[1, 2] == pytest.approx(values[0, 2], rel=0.05)

    mpiClass.running(checking,torchdevice, mpiObjectInput=mpi)


