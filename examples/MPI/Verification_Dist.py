
from lettuce import mpiClass
import lettuce as lt
import torch
import sys
import numpy as np

import matplotlib.pyplot as plt

def run(device, mpiObject):
    resolution = 50
    steps = 100
    reporterIntervall=10
    print(f"3D multi core test TGV, device: {device}, resolution: {resolution}, steps: {steps}")
    lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float64,mpiObject=mpiObject)  # single precision - float64 for double precision
    
    flow = lt.TaylorGreenVortex3D(resolution, 1600, 0.1, lattice)

    # select collision model - try also KBCCollision or RegularizedCollision
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow, lattice, collision, streaming)
    
    #testing all observables
    
    rep = lt.VTKReporter(lattice, flow, steps-1, "./results/vtk")
    #simulation.reporters.append(rep)
    
    maxVel=lt.MaximumVelocity(lattice,flow,mpiObject)
    maxVelReporter=lt.ObservableReporter(maxVel,interval=reporterIntervall, out=None)
    simulation.reporters.append(maxVelReporter)

    incompressibleKineticEnergy=lt.IncompressibleKineticEnergy(lattice,flow)
    incompressibleKineticEnergyReporter=lt.ObservableReporter(incompressibleKineticEnergy,interval=reporterIntervall, out=None)
    simulation.reporters.append(incompressibleKineticEnergyReporter)

    
    enstrophy = lt.Enstrophy(lattice, flow)
    enstrophyReporter = lt.ObservableReporter(enstrophy, interval=reporterIntervall, out=None)
    simulation.reporters.append(enstrophyReporter)

    mass=lt.Mass(lattice, flow)
    massReporter = lt.ObservableReporter(mass, interval=reporterIntervall, out=None)
    simulation.reporters.append(massReporter)

    #steprep=StepReporter(lattice,flow,interval=25,mpiObj=mpiObject)
    #simulation.reporters.append(steprep)


    if mpiObject.getRank() == 0:
        lattice2 = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float64)
        flow2 = lt.TaylorGreenVortex3D(resolution, 1600, 0.1, lattice)
        collision2 = lt.BGKCollision(lattice2, tau=flow2.units.relaxation_parameter_lu)
        streaming2 = lt.StandardStreaming(lattice2)
        simulation2 = lt.Simulation(flow2, lattice2, collision2, streaming2)
        rep2 = lt.VTKReporter(lattice2, flow2, steps-1, "./results/vtk2")
        #simulation2.reporters.append(rep2)
        
        maxVel2=lt.MaximumVelocity(lattice,flow)
        maxVelReporter2=lt.ObservableReporter(maxVel2,interval=reporterIntervall, out=None)
        simulation.reporters.append(maxVelReporter2)

        incompressibleKineticEnergy2=lt.IncompressibleKineticEnergy(lattice,flow)
        incompressibleKineticEnergyReporter2=lt.ObservableReporter(incompressibleKineticEnergy2,interval=reporterIntervall, out=None)
        simulation.reporters.append(incompressibleKineticEnergyReporter2)

        
        enstrophy2 = lt.Enstrophy(lattice, flow)
        enstrophyReporter2 = lt.ObservableReporter(enstrophy2, interval=reporterIntervall, out=None)
        simulation.reporters.append(enstrophyReporter2)

        mass2=lt.Mass(lattice, flow)
        massReporter2 = lt.ObservableReporter(mass2, interval=reporterIntervall, out=None)
        simulation.reporters.append(massReporter2)

    for i in range(0, steps):
       
        simulation.step(1)
        
        #rank 0 bekommt ein gird was komplet ist nur für statistik genutzt
        #f = flow.grid.reassemble(simulation.f)
        
        if mpiObject.getRank() == 0:
            #unterschied zwischen dem jetzigen berechneten und local berechneten (unterschied zwischen verteilten rechnen und localen rechnen)
            
            
            simulation2.step(1)
            #f2 = simulation2.f
            #print("{:.20f}".format(torch.max(torch.abs(f-f2))))

    #done with simulating
    if mpiObject.getRank() == 0:
        maxVelD = np.asarray(maxVelReporter.out)
        maxVelS= np.asarray(maxVelReporter2.out)

        kineticD = np.asarray(incompressibleKineticEnergyReporter.out)
        kineticS= np.asarray(incompressibleKineticEnergyReporter2.out)

        enstD = np.asarray(enstrophyReporter.out)
        enstS= np.asarray(enstrophyReporter2.out)
        
        massD = np.asarray(massReporter.out)
        massS= np.asarray(massReporter2.out)

        plt.figure()
        plt.tight_layout()
        
        plt.figure(figsize=(10, 10))
        plt.subplots_adjust(hspace = 0.501)
        plt.title("MaxVelocity")
        plt.subplot(311)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, maxVelD.shape[0])), maxVelD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Verteilt MaxVelocity")

        plt.subplot(312)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, maxVelS.shape[0])), maxVelS[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Erwarted MaxVelocity")

        plt.subplot(313)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, maxVelS.shape[0])), maxVelS[:, 2]-maxVelD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Diff-MaxVelocity")
        plt.show()
        
        plt.show()
        plt.savefig(fname="MaxVelocity")

        plt.figure()
        plt.title("incompressibleKineticEnergy")
        plt.subplot(311)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, kineticD.shape[0])), kineticD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Verteilt incompressibleKineticEnergy")

        plt.subplot(312)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, kineticS.shape[0])), kineticS[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Erwarted incompressibleKineticEnergy")

        plt.subplot(313)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, kineticD.shape[0])), kineticS[:, 2]-kineticD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Diff-incompressibleKineticEnergy")
        plt.savefig(fname="incompressibleKineticEnergy")

        plt.figure()
        plt.title("Enstrophy")
        plt.subplot(311)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, enstD.shape[0])), enstD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Verteilt Enstrophy")

        plt.subplot(312)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, enstS.shape[0])), enstS[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Erwarted Enstrophy")
        plt.savefig(fname="Enstrophy")

        plt.subplot(313)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, enstS.shape[0])), enstS[:, 2]-enstD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Diff-Enstrophy")
        plt.savefig(fname="Enstrophy")

        plt.figure()
        plt.title("Mass")
        plt.subplot(311)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, massD.shape[0])), massD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Verteilt Mass")

        plt.subplot(312)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, massS.shape[0])), massS[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Erwarted Mass")
        plt.savefig(fname="Mass")

        plt.subplot(313)
        plt.plot(simulation.flow.units.convert_time_to_pu(range(0, massS.shape[0])), massS[:, 2]-massD[:, 2])
        plt.xlabel("Time")
        plt.ylabel("Diff-Mass")
        plt.savefig(fname="Mass")

        print("done")

if __name__ == "__main__":
    device = torch.device("cuda")
    
    
    pcList=[[20,"hpc-node000"]]
    gpuList=[[2,"gpu-node009",50]]
    mpiOBJ=lt.mpiObject(1,gpuList=gpuList)
    lt.running(run,device,mpiOBJ)
