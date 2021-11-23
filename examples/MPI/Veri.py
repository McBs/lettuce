
from lettuce import mpiClass
import lettuce as lt
import torch
import sys
import numpy as np

import matplotlib.pyplot as plt

def run(device, mpiObject):
    resolution = 128
    steps = 20000
    reporterIntervall=10
    dtype=torch.float64
    print(f"3D multi core test TGV, device: {device}, resolution: {resolution}, steps: {steps}")
    lattice = lt.Lattice(lt.D3Q27, device=device, dtype=dtype,MPIObject=mpiObject)  # single precision - float64 for double precision
    
    flow = lt.TaylorGreenVortex3D(resolution, 1600, 0.1, lattice)

    # select collision model - try also KBCCollision or RegularizedCollision
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow, lattice, collision, streaming)
    
    #testing all observables
    
    rep = lt.VTKReporter(lattice, flow, steps-1, "./results/vtk")
    #simulation.reporters.append(rep)
    
    maxVel=lt.MaximumVelocity(lattice,flow)
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

    steprep=lt.StepReporter(lattice,flow ,interval=reporterIntervall)
    simulation.reporters.append(steprep)


    if mpiObject.rank == 0:
        lattice2 = lt.Lattice(lt.D3Q27, device=device, dtype=dtype)
        flow2 = lt.TaylorGreenVortex3D(resolution, 1600, 0.1, lattice2)
        collision2 = lt.BGKCollision(lattice2, tau=flow2.units.relaxation_parameter_lu)
        streaming2 = lt.StandardStreaming(lattice2)
        simulation2 = lt.Simulation(flow2, lattice2, collision2, streaming2)
        rep2 = lt.VTKReporter(lattice2, flow2, steps-1, "./results/vtk2")
        #simulation2.reporters.append(rep2)
        
        maxVel2=lt.MaximumVelocity(lattice2,flow2)
        maxVelReporter2=lt.ObservableReporter(maxVel2,interval=reporterIntervall, out=None)
        simulation2.reporters.append(maxVelReporter2)

        incompressibleKineticEnergy2=lt.IncompressibleKineticEnergy(lattice2,flow2)
        incompressibleKineticEnergyReporter2=lt.ObservableReporter(incompressibleKineticEnergy2,interval=reporterIntervall, out=None)
        simulation2.reporters.append(incompressibleKineticEnergyReporter2)

        
        enstrophy2 = lt.Enstrophy(lattice2, flow2)
        enstrophyReporter2 = lt.ObservableReporter(enstrophy2, interval=reporterIntervall, out=None)
        simulation2.reporters.append(enstrophyReporter2)

        mass2=lt.Mass(lattice2, flow2)
        massReporter2 = lt.ObservableReporter(mass2, interval=reporterIntervall, out=None)
        simulation2.reporters.append(massReporter2)

        steprep2=lt.StepReporter(lattice2,flow2 ,interval=reporterIntervall)
        simulation2.reporters.append(steprep2)
       
    print("perf ",simulation.step(steps))
        
        #rank 0 bekommt ein gird was komplet ist nur für statistik genutzt
        #f = flow.grid.reassemble(simulation.f)
    print("P parallel Done")
    if mpiObject.rank == 0:
            #unterschied zwischen dem jetzigen berechneten und local berechneten (unterschied zwischen verteilten rechnen und localen rechnen)
            
        print("S parallel Done")
        simulation2.step(steps)
            #f2 = simulation2.f
            #print("{:.20f}".format(torch.max(torch.abs(f-f2))))

    #done with simulating
    if mpiObject.rank == 0:
        maxVelD = np.asarray(maxVelReporter.out)
        maxVelS= np.asarray(maxVelReporter2.out)

        kineticD = np.asarray(incompressibleKineticEnergyReporter.out)
        kineticS= np.asarray(incompressibleKineticEnergyReporter2.out)

        enstD = np.asarray(enstrophyReporter.out)
        enstS= np.asarray(enstrophyReporter2.out)
        
        massD = np.asarray(massReporter.out)
        massS= np.asarray(massReporter2.out)

        if(False):
            import pickle
            filenameu = f'./data/Ver/maxVELD.p'
            outfileu = open(filenameu,'wb')
            pickle.dump(maxVelD,outfileu)
            filenameu = f'./data/Ver/maxVELs.p'
            outfileu = open(filenameu,'wb')
            #pickle.dump(maxVelS,outfileu)
            filenameu = f'./data/Ver/kineticD.p'
            outfileu = open(filenameu,'wb')
            pickle.dump(kineticD,outfileu)
            filenameu = f'./data/Ver/kinetics.p'
            outfileu = open(filenameu,'wb')
            #pickle.dump(kineticS,outfileu)
            filenameu = f'./data/Ver/enstD.p'
            outfileu = open(filenameu,'wb')
            pickle.dump(enstD,outfileu)
            filenameu = f'./data/Ver/ensts.p'
            outfileu = open(filenameu,'wb')
            #pickle.dump(enstS,outfileu)
            filenameu = f'./data/Ver/massD.p'
            outfileu = open(filenameu,'wb')
            pickle.dump(massD,outfileu)
            filenameu = f'./data/Ver/massS.p'
            outfileu = open(filenameu,'wb')
            #pickle.dump(massS,outfileu)
            
        if(True):
            last=len(kineticS)-1
            print("Shape: ",maxVelD.shape)
            print("Last Max Velocity: Expected: ", maxVelS[last,2], " Got: ",maxVelD[last,2])
            print("Last incompressibleKineticEnergy: Expected: ", kineticS[last,2], " Got: ",kineticD[last,2])
            print("Last Enstrophy: Expected: ", enstS[last,2], " Got: ",enstD[last,2])
            print("Last Mass: Expected: ", massS[last,2], " Got: ",massD[last,2])
        

        print("done")

if __name__ == "__main__":
    device = torch.device("cuda")
    
    
    pcList=[[20,"hpc-node000"]]
    gpuList=[[4,"gpu-node001"],[4,"gpu-node002"],[4,"gpu-node003"],[4,"gpu-node004"],[1,"gpu-node005"],[1,"gpu-node006"],[1,"gpu-node007"],[1,"gpu-node008"],[2,"gpu-node009"],[2,"gpu-node010"]]
    mpiOBJ=lt.mpiObject(1,gpuList=gpuList,distributefromRank0=0)
    lt.running(run,device,mpiOBJ)