import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
init_f_neq=True
def run(device, mpiObject):
    print("start")

    """Use Taylor Green 2D for convergence test in diffusive scaling."""
    
    lattice = lt.Lattice(lt.D2Q9, device, dtype=torch.float32)
    error_u_old = None
    error_p_old = None
    print(("{:>15} " * 5).format("resolution", "error (u)", "order (u)", "error (p)", "order (p)"))

    for i in range(4, 27):
        resolution = 2 ** i
        mach_number = 8 / resolution

        # Simulation
        flow = lt.TaylorGreenVortex2D(resolution=resolution, reynolds_number=10000, mach_number=mach_number,
                                   lattice=lattice,mpiObject=mpiObject)
        collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
        streaming = lt.StandardStreaming(lattice,mpiObject=mpiObject)
        simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming,mpiObject=mpiObject)
        if init_f_neq:
            simulation.initialize_f_neq()
        error_reporter = lt.ErrorReporter(lattice, flow, interval=1, out=None,mpiObj=mpiObject)
        simulation.reporters.append(error_reporter)
        for _ in range(10 * resolution):
            simulation.step(1)
        if(mpiObject.rank==0):
            error_u, error_p = np.mean(np.abs(error_reporter.out), axis=0).tolist()
            factor_u = 0 if error_u_old is None else error_u_old / error_u
            factor_p = 0 if error_p_old is None else error_p_old / error_p
            error_u_old = error_u
            error_p_old = error_p
            filenameu = f'./data/error/feq/error_u_{resolution}_float32.p'
            outfileu = open(filenameu,'wb')
            pickle.dump(error_u,outfileu)
            filenamep = f'./data/error/feq/error_p_{resolution}_float32.p'
            outfilep = open(filenamep,'wb')
            pickle.dump(error_p,outfilep)
            print("{:15} {:15.2e} {:15.1f} {:15.2e} {:15.1f}".format(
                resolution, error_u, factor_u / 2, error_p, factor_p / 2))
    if factor_u / 2 < 1.9:
        print("Velocity convergence order < 2.")
    if factor_p / 2 < 0.9:
        print("Velocity convergence order < 1.")
    if factor_u / 2 < 1.9 or factor_p / 2 < 0.9:
        print("Sys Exit")

    print("---------------------------------------------------------------------------")
    
   
    print("Python end")
    # ---------- Plot kinetic energy over time (PU) -------------
    # grab output of kinetic energy reporter
   

if __name__ == "__main__":
    device = torch.device("cuda")
    pcList=[[0,20],[1,40]]
    gpuList=[[4,"gpu-node001"],[4,"gpu-node002"],[4,"gpu-node003"],[4,"gpu-node004"],[1,"gpu-node005"],[1,"gpu-node006"],[1,"gpu-node007"],[1,"gpu-node008"],[2,"gpu-node009"],[2,"gpu-node010"]]
    mpiOBJ=lt.mpiObject(0,nodeList=pcList,gpuList=gpuList,setParts=0,gridRefinment=0)
    lt.running(run,device,mpiOBJ)
    