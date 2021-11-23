import lettuce as lt
import torch

def run(device, mpiObject):
    #  Set up simulation 
    lattice = lt.Lattice(lt.D3Q19, device=device, dtype=torch.float32,MPIObject=mpiObject)  # single precision - float64 for double precision
    resolution = 128  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
    flow = lt.TaylorGreenVortex3D(resolution=resolution, reynolds_number=1600, mach_number=0.05, lattice=lattice)
   
    # select collision model 
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow, lattice, collision, streaming)

    #  Simulate until time = 10 (PU) 
    x=simulation.flow.units.convert_time_to_lu(10)
    print("Simulating", x, "steps.")

    # runs simulation, but also returns overall performance in MLUPS (million lattice units per second)
    print("MLUPS: ", simulation.step(int(x)))
    

if __name__ == "__main__":

    device = torch.device("cpu") #setting up target device if Lettuce should use GPUs change "cpu" to "cuda"
    sizeList=[[0,20],[1,40]]     #Redifine what rank does what Slize. Here Rank 0 covers from 0 to 20 and Rank 1 covers from 20 to 60
    gpuList=[[4,"gpuNode001"],[1,"gpuNode002"],[1,"gpuNode003"],[2,"gpuNode004"]] # Setting up the GPU List for automatic Destribution of GPUs. First number defines how many GPUs are present followed by the name of the computer
    mpiOBJ=lt.mpiObject(1,sizeList=sizeList,gpuList=gpuList,setParts=0,distributefromRank0=1,initOnCPU=1,gridRefinment=0,printUserInfo=1) #creating the MPI-Object
    lt.running(run,device,mpiOBJ)
    