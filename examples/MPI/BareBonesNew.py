import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

def run(device, mpiObject):
    print("start")

    # ---------- Set up simulation -------------
    lattice = lt.Lattice(lt.D2Q9, device=device, dtype=torch.float64,MPIObject=mpiObject)  # single precision - float64 for double precision
    resolution = 120  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
    flow = lt.TaylorGreenVortex2D(resolution=resolution, reynolds_number=10000, mach_number=0.05, lattice=lattice)
    print(flow.units.relaxation_parameter_lu)
    # select collision model - try also KBCCollision or RegularizedCollision
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow, lattice, collision, streaming)
    VTKreport = lt.VTKReporter(lattice, flow, interval=100,filename_base="./data/GRE/Eoutput")
   
    #simulation.initialize_f_neq()  # initialize non-conserved moments
    simulation.reporters.append(VTKreport)
   
   
    # ---------- Simulate until time = 10 (PU) -------------
    print("Simulating", int(simulation.flow.units.convert_time_to_lu(10)), "steps! Maybe drink some water in the meantime.")
    # runs simulation, but also returns overall performance in MLUPS (million lattice units per second)

    x=simulation.flow.units.convert_time_to_lu(10)
    print("MLUPS: ", simulation.step(int(x)))
    
    print("Python end")
    # ---------- Plot kinetic energy over time (PU) -------------
    # grab output of kinetic energy reporter
   

if __name__ == "__main__":
    device = torch.device("cpu")
    pcList=[[0,20],[1,40]]
    gpuList=[[4,"gpu-node001"],[4,"gpu-node002"],[4,"gpu-node003"],[4,"gpu-node004"],[1,"gpu-node005"],[1,"gpu-node006"],[1,"gpu-node007"],[1,"gpu-node008"],[2,"gpu-node009"],[2,"gpu-node010"]]
    mpiOBJ=lt.mpiObject(0,sizeList=pcList,gpuList=gpuList,setParts=0,gridRefinment=0,printUserInfo=1)
    lt.running(run,device,mpiOBJ)
    
