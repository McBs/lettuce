import lettuce as lt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import warnings
import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet, InterpolatedBounceBackBoundary, SlipBoundary
# from lettuce.flows.cylinder2D import Cylinder2D
from lettuce.flows.obstacleCylinder import ObstacleCylinder

from lettuce.max import draw_circular_mask

import torch
import time
import datetime
import os
import psutil
import shutil
from pyevtk.hl import imageToVTK
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
warnings.simplefilter("ignore")

##################################################
#ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0")
parser.add_argument("--gpd", default=20, type=int, help="number of gridpoints per diameter")
parser.add_argument("--dpx", default=0, type=int, help="domain length in diameters")
parser.add_argument("--dpy", default=19, type=int, help="domain height in diameters")
parser.add_argument("--dpz", default=0, type=float, help="domain width in diameters")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to dimulate")
parser.add_argument("--collision", default="bgk", help="collision operator (bgk, kbc, reg)")
parser.add_argument("--lateral_walls", default='periodic', help="boundary condition in y direction (periodic, bounceback, slip)")
parser.add_argument("--bc_type", default='fwbb', help="bounce back algorithm (fwbb, hwbb, ibb1)")
parser.add_argument("--name", default="2Dcylinder", help="name of the simulation, appears in output directory name")
parser.add_argument("--stencil", default="D2Q9", help="stencil (D2Q9, D3Q27, D3Q19, D3Q15)")
parser.add_argument("--output_vtk", default=False, type=bool, help="bool, output vtk-data with 10 fps (large!)")
parser.add_argument("--device", default="cuda:0", help="cuda-device for multi-gpu nodes cuda:[0,1,2,3]")
parser.add_argument("--calcUProfiles", default=False, help="calculate average velocity profiles as in Di Ilio et al. 2018 and output plots and time-averages data for plots")
parser.add_argument("--outputUProfiles", default=False, help="output average velocity profiles over time (full timeseries)")
parser.add_argument("--nan_reporter", default=False, help="stop simulation if NaN is detected in f field")
parser.add_argument("--from_cpt", default=False, help="start from checkpoint. (!) provide --cpt_file path")
parser.add_argument("--cpt_file", default=None, help="path and name of cpt_file to use if --from_cpt=True")
parser.add_argument("--sim_i", default=0, type=int, help="step index of last checkpoints-step to start from for time-indexing of observables etc.")
parser.add_argument("--write_cpt", default=False, help="write checkpoint after finishing simulation")


args = vars(parser.parse_args())

##################################################
#PARAMETERS

re = args["re"]            # Reynoldszahl
Ma = 0.05           # Machzahl
n_steps = args["n_steps"]    # Schrittzahl
setup_diameter = 1  # D_PU = char_length_pu -> this defines the PU-Reference
flow_velocity = 1   # U_PU = char_velocity_pu -> this defines the PU-Reference velocity (u_max of inflow)

periodic_start = 0.9  # relative start of peak_finding for Cd_mean Measurement to cut of any transients

gridpoints_per_diameter = args["gpd"]  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
domain_height_in_D = args["dpy"]  # D/Y  -> this defines the domain-size and total number of Lattice-Nodes
if args["dpx"] == 0:
    domain_length_in_D = 2 * domain_height_in_D  # D/X = domain length in X- / flow-direction
else:
    domain_length_in_D = args["dpx"]
domain_width_in_D = args["dpz"] 
if args["dpz"] != 0:
    print("you are providing a dpz value. This is a 2D simulation! dpz is discarded")

# if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
# ...if DpY is even, GPD will be corrected to even GPD for symemtrical cylinder
# ...use odd DpY to use odd GPD
gpd_correction=False
if domain_height_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
    gpd_correction = True   # gpd_was_corrected-flag
    gpd_setup = gridpoints_per_diameter   # store old gpd for output
    gridpoints_per_diameter = int(gridpoints_per_diameter/2)*2   # make gpd even
    print("(!) domain_height_in_D (DpY) is even, gridpoints_per_diameter will be "+str(gridpoints_per_diameter)+". Use odd domain_height_in_D to enable use of odd GPD!")


# OVERWRITE n_steps, if t_target is given
T_target = 140
if args["t_target"] > 0:
    T_target = args["t_target"]
    n_steps = int(T_target*((gridpoints_per_diameter)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))))

# SIMULATOR settings
u_init = 0    # initial velocity field: # 0: uniform u=0, # 1: uniform u=1, # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)
perturb_init = True   # perturb initial symmetry by small sine-wave in initial velocity field -> triggers Karman-vortex street for Re > 46
lateral_walls=args["lateral_walls"]
bc_type=args["bc_type"]
vtk_fps = 10    # FramesPerSecond (PU) for vtk-output
cuda_device = args["device"]
nan_reporter = args["nan_reporter"]

gridpoints = gridpoints_per_diameter**2*domain_length_in_D*domain_height_in_D # calc. total number of gridpoints

##################################################
# DATA OUTPUT SETTINGS (observables, stats and vtk)

output_data = True  # output/log parameters, observables and vtk or vti (if output_vtk=True)
if args["output_vtk"]:
    output_vtk=True  # is overwritten by output_data=False (see below)
else:
    output_vtk=False

from_cpt=False
if args["from_cpt"]:  # if sim should start from checkpoints file
    from_cpt=True
    cpt_file = args["cpt_file"]
    cpt_sim_i = int(args["sim_i"])
write_cpt=False
if args["write_cpt"]:
    write_cpt=True

if output_data:  # only calculate u-profiles if data should be saved
    calculate_velocity_profile = args["calcUProfiles"]
    if calculate_velocity_profile:  # only output u-profiles, if they are calculated
        output_velocity_profile = args["outputUProfiles"]
    else:
        output_velocity_profile = False
else:
    calculate_velocity_profile = False
    output_velocity_profile = False

# naming: specify name/number and parameters to put in directory- and datafile-names
name = args["name"]

if output_data:  # toggle output of parameters, observables and vti/vtk files
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%y%m%d")+"_"+timestamp.strftime("%H%M%S")

    # specify output directory/path
    #output_path = "/mnt/ScratchHDD1/Max_Scratch/lbm_simulations"  # lokal HBRS
    #output_path = "/home/max/Documents/lbm_simulations"  # lokal Bonn
    output_path = "/home/mbille3s/02_lbm_simulations"  # cluster HBRS
    scratch_dir = "/scratch/mbille3s/21_LBM/01_data"  # cluster HBRS for vti-output of big simulations
    dir_name = "/data_" + str(timestamp) + "_" + name  # create directory name for all outputs to be saved in
    os.makedirs(output_path+dir_name)

    # specify input directory/path for rerefence data
    #diIlio_path = '/home/mbille/lettuce/myTest/DiIlio_Fig09_data/'  # likal HBRS
    diIlio_path = '/scratch/mbille3s/21_LBM/03_reference_data/DiIlio_2018/'  # cluster HBRS

    if calculate_velocity_profile:
        os.makedirs(output_path+dir_name+"/AvgVelocity_Data")
    
    print("dir_name: "+dir_name)
    if output_vtk:
        #vtk_path = output_path+dir_name+"/vtk/out"  # subdirectory for vtk/vti output
        os.makedirs(scratch_dir+dir_name+"/vtk/")
        vtk_path = scratch_dir + dir_name + "/vtk/out"
        print("vtk_path: " + vtk_path)
    elif write_cpt:
        os.makedirs(scratch_dir + dir_name)
else:  # "no output" suppresses the vtk output too
    output_vtk = False

##################################################
#SIM SETUP (instatiate objects, calculate&place obstacle, append reporters)

# lattice
lattice = lt.Lattice(lt.D2Q9, cuda_device, dtype=torch.float64)
stencil_choice = "D2Q9"

# flow
flow = ObstacleCylinder(shape=(domain_length_in_D*gridpoints_per_diameter,
                               domain_height_in_D*gridpoints_per_diameter),
                        reynolds_number=re, mach_number=Ma,
                        lattice=lattice,
                        char_length_pu=setup_diameter,
                        char_length_lu=gridpoints_per_diameter,  ###IMPORTANT D=GPD
                        char_velocity_pu=flow_velocity,
                        lateral_walls=args["lateral_walls"],
                        bc_type=args["bc_type"],
                        perturb_init=perturb_init,
                        u_init=u_init
                        )

### Simulation-Object (Simulator/solver) and additional settings (tau, collision operator)
tau = flow.units.relaxation_parameter_lu
re_g = flow.units.characteristic_velocity_lu/(lattice.stencil.cs**2 * (tau-0.5))  # grid reynolds number (should be O(10))
print("Re_g = ", re_g)

# collision operator
if args["collision"] == "kbc":
    collision = lt.KBCCollision2D(lattice,tau)
    collision_choice ="kbc"
elif args["collision"] == "reg":
    collision = lt.RegularizedCollision(lattice, tau)
    collision_choice ="reg"
else:
    collision = lt.BGKCollision(lattice, tau)
    collision_choice ="bgk"

# solver
sim = lt.Simulation(flow, lattice, 
                    collision,
                    # lt.BGKCollision(lattice, tau),
                    # lt.RegularizedCollision(lattice, tau), 
                    # lt.KBCCollision2D(lattice,tau),
                    lt.StandardStreaming(lattice)
                    )
### Reporter

# VTK Reporter -> visualization
if output_vtk == True:
    VTKreport = lt.VTKReporter(lattice, flow, interval=int(flow.units.convert_time_to_lu(1/vtk_fps)), filename_base=vtk_path)
    sim.reporters.append(VTKreport)
    # export obstacle
    mask_dict = dict()
    mask_dict["mask"] = flow.obstacle_mask[...,None].astype(int)
    imageToVTK(
        path=scratch_dir+dir_name+"/vtk/obstacle_point",
        pointData=mask_dict,
    )
    imageToVTK(
        path=scratch_dir+dir_name+"/vtk/obstacle_cell",
        cellData=mask_dict,
    )

# Observable reporter: drag coefficient
DragObservable = lt.DragCoefficient(lattice,flow,sim._boundaries[-1],area=setup_diameter)  # create observable // ! area A=2*r is in PU
Dragreport = lt.ObservableReporter(DragObservable, out=None)  # create reporter and link to created observable
sim.reporters.append(Dragreport)  # append reporter to reporter-list of simulator/solver
    
# Observable reporter: lift coefficient
LiftObservable = lt.LiftCoefficient(lattice,flow,sim._boundaries[-1],area=setup_diameter)
Liftreport = lt.ObservableReporter(LiftObservable, out=None)
sim.reporters.append(Liftreport)

# AvgVelocityReporter
if calculate_velocity_profile:
    # define positions
    position_1 = flow.x_pos-0.5 + 1.06 * flow.radius * 2  #int(round(flow.x_pos + 1.06 * flow.radius * 2 , 0))
    position_2 = flow.x_pos-0.5 + 1.54 * flow.radius * 2  #int(round(flow.x_pos + 1.54 * flow.radius * 2 , 0))
    position_3 = flow.x_pos-0.5 + 2.02 * flow.radius * 2  #int(round(flow.x_pos + 2.02 * flow.radius * 2 , 0))
    print("V_avg positions:" + " p1: " + str(position_1) + " p2:  " + str(position_2) + " p3:  " + str(position_3))

    # create and append AvgVelocity-reporter
    AvgVelocity1 = lt.AverageVelocityReporter(lattice,flow, position_1)
    sim.reporters.append(AvgVelocity1)
    AvgVelocity2 = lt.AverageVelocityReporter(lattice,flow, position_2)
    sim.reporters.append(AvgVelocity2)
    AvgVelocity3 = lt.AverageVelocityReporter(lattice,flow, position_3)
    sim.reporters.append(AvgVelocity3)

# NaN STOP
if nan_reporter:
    NaNReporter = lt.NaNReporter(flow, lattice, n_steps, T_target)
    sim.reporters.append(NaNReporter)


##################################################
#PRINT PARAMETERS prior to simulation:
print("shape_LU:", gridpoints_per_diameter*domain_length_in_D, "x", gridpoints_per_diameter*domain_height_in_D)
print("T with", n_steps, "steps:", round(n_steps * (setup_diameter/(gridpoints_per_diameter))*(Ma*1/np.sqrt(3)/flow_velocity),2), "seconds")
print("n_steps to simulate 1 second:", round(((gridpoints_per_diameter)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))),2), "steps")
print("n_steps to simulate",T_target,"seconds:",T_target*round(((gridpoints_per_diameter)/setup_diameter)*(flow_velocity/(Ma*1/np.sqrt(3))),2), "steps")
if output_vtk:
    print("generates approx.", int(vtk_fps*(n_steps * (setup_diameter/(gridpoints_per_diameter))*(Ma*1/np.sqrt(3)/flow_velocity)))+1, ".vti/.vtk-frames")

##################################################
### export parameters to file

if output_data:
    output_file = open(output_path+dir_name+"/"+timestamp + "_parameters.txt", "a")
    output_file.write("DATA for "+timestamp)
    output_file.write("\n\n###   SIM-Parameters   ###")
    output_file.write("\nRe = "+str(re))
    output_file.write("\nn_steps = "+str(n_steps))
    output_file.write("\nT_target = "+str(flow.units.convert_time_to_pu(n_steps))+" seconds")
    output_file.write("\ngridpoints_per_diameter (gpd) = "+str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: "+str(gpd_setup)+" to "+str(gridpoints_per_diameter)+" because D/Y is even")
    output_file.write("\nDpX (D/X) = " + str(domain_length_in_D))
    output_file.write("\nDpY (D/Y) = "+str(domain_height_in_D))
    if lattice.D == 3:
        output_file.write("\nDpZ (D/Z) = "+str(domain_width_in_D))
    output_file.write("\nshape_LU: "+ str(flow.shape))
    output_file.write(("\ntotal_number_of_gridpoints: "+str(lattice.rho(sim.f).numel())))
    output_file.write("\nbc_type = "+str(bc_type))
    output_file.write("\nlateral_walls = "+str(lateral_walls))
    output_file.write("\nstencil = "+str(stencil_choice))
    output_file.write("\ncollision = " + str(collision_choice))
    output_file.write("\n")
    output_file.write("\nMa = " + str(Ma))
    output_file.write("\ntau = " + str(tau))
    output_file.write("\ngrid_reynolds_number (Re_g) = " + str(re_g))
    output_file.write("\n")
    output_file.write("\nsetup_diameter_PU = " + str(setup_diameter))
    output_file.write("\nflow_velocity_PU = " + str(flow_velocity))
    output_file.write("\nu_init = " + str(u_init))
    output_file.write("\nperturb_init = " + str(perturb_init))
    output_file.write("\n")
    output_file.write("\noutput_vtk = " + str(output_vtk))
    output_file.write("\nvtk_fps = " + str(vtk_fps))
    output_file.close()

### calculate and export 2D obstacle_mask as .png
draw_circular_mask(lattice, gridpoints_per_diameter, output_data=output_data, filebase=output_path+dir_name, print_data=False)

##################################################
# LOAD CHECKPOINT and ajust sim.i
if from_cpt:
    if cpt_file is not None:
        sim.load_checkpoint(cpt_file, device=lattice.device)
        #sim.f = sim.f.to(lattice.device, copy=False)  # if device is different to device used on previous simulation!
        sim.i=cpt_sim_i

##################################################
# RUN SIMULATION

t_start=time.time()

mlups = sim.step(int(n_steps)) # Simulation with n_steps steps

t_end=time.time()
runtime=t_end-t_start
# output stats
print("MLUPS:", mlups)
print("PU-Time: ",flow.units.convert_time_to_pu(n_steps)," seconds")
print("number of steps:",n_steps)
print("runtime: ",runtime, "seconds (", round(runtime/60,2),"minutes )")

c_time = sim.time_avg["time_collision"]
fc_time = sim.time_avg["time_store_f_collided"]
s_time = sim.time_avg["time_streaming"]
b_time = sim.time_avg["time_boundary"]
r_time = sim.time_avg["time_reporter"]
sum_time = sim.time_avg["time_collision"] + sim.time_avg["time_store_f_collided"] + sim.time_avg["time_streaming"] + sim.time_avg["time_boundary"] + sim.time_avg["time_reporter"]

print("collision avg. time:", sim.time_avg["time_collision"], "seconds (" + str(round(100 * c_time/sum_time, 2)) + " %)")
print("f_c_store avg. time:", sim.time_avg["time_store_f_collided"], "seconds (" + str(round(100 * fc_time/sum_time, 2)) + " %)")
print("streaming avg. time:", sim.time_avg["time_streaming"], "seconds (" + str(round(100 * s_time/sum_time, 2)) + " %)")
print("boundary avg. time:", sim.time_avg["time_boundary"], "seconds (" + str(round(100 * b_time/sum_time, 2)) + " %)")
print("reporter avg. time:", sim.time_avg["time_reporter"], "seconds (" + str(round(100 * r_time/sum_time, 2)) + " %)")


print("current VRAM (MB): ", torch.cuda.memory_allocated(lattice.device)/1024/1024)
print("max. VRAM (MB): ", torch.cuda.max_memory_allocated(lattice.device)/1024/1024)

[cpuLoad1,cpuLoad5,cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU % avg. over last 1 min, 5 min, 15 min; ", round(cpuLoad1,2), round(cpuLoad5,2), round(cpuLoad15,2))

ram = psutil.virtual_memory()
print("current total RAM usage [MB]: " + str(round(ram.used/(1024*1024),2)) + " of " + str(round(ram.total/(1024*1024),2)) + " MB")

### export stats
if output_data:
    output_file = open(output_path+dir_name+"/"+timestamp + "_stats.txt", "a")
    output_file.write("DATA for "+timestamp)
    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = "+str(runtime)+ " seconds (="+str(runtime/60)+" minutes)")
    output_file.write("\nMLUPS = "+str(mlups))
    output_file.write("\n")
    output_file.write("\navg. Collision-Time [s] = " + str(c_time) + " (" + str(round(100 * c_time/sum_time, 2)) + " %)")
    output_file.write("\navg. store_fcl-Time [s] = " + str(fc_time) + " (" + str(round(100 * fc_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Streaming-Time [s] = " + str(s_time) + " (" + str(round(100 * s_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Boundary-Time  [s] = " + str(b_time) + " (" + str(round(100 * b_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Reporter-Time  [s] = " + str(r_time) + " (" + str(round(100 * r_time/sum_time, 2)) + " %)")
    output_file.write("\n")
    output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(lattice.device)/1024/1024))
    output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(lattice.device)/1024/1024))
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\n")
    output_file.write("\ntotal current RAM usage [MB]: " + str(round(ram.used/(1024*1024),2)) + " of " + str(round(ram.total/(1024*1024),2)) + " MB")
    output_file.close()

##################################################
# WRITE CHECKPOINT
if write_cpt:
    try:
        sim.save_checkpoint(scratch_dir+dir_name+"/"+timestamp + "_f_"+str(sim.i)+".cpt", device="cpu")
    except:
        print("(!) writing checkpoint failed...")

##################################################
# CREATE OBSERVABLE-PLOTS & SAVE OBSERVABLE-timeseries

# Avg VELOCITY
if calculate_velocity_profile:
    avg_u_start = 0.5

    # import reference data: (data is: first collumn Y/D, second column u_d/u_char)
    # ux
    p1_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_LS1993.csv', delimiter=';')
    p2_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_LS1993.csv', delimiter=';')
    p3_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_LS1993.csv', delimiter=';')

    p1_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_KM2000.csv', delimiter=';')
    p2_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_KM2000.csv', delimiter=';')
    p3_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_KM2000.csv', delimiter=';')

    p1_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_WR2008.csv', delimiter=';')
    p2_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_WR2008.csv', delimiter=';')
    p3_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_WR2008.csv', delimiter=';')

    p1_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_DI2018.csv', delimiter=';')
    p2_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_DI2018.csv', delimiter=';')
    p3_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_DI2018.csv', delimiter=';')

    # uy
    p1_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_LS1993.csv', delimiter=';')
    p2_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_LS1993.csv', delimiter=';')
    p3_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_LS1993.csv', delimiter=';')

    p1_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_KM2000.csv', delimiter=';')
    p2_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_KM2000.csv', delimiter=';')
    p3_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_KM2000.csv', delimiter=';')

    p1_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_WR2008.csv', delimiter=';')
    p2_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_WR2008.csv', delimiter=';')
    p3_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_WR2008.csv', delimiter=';')

    p1_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_DI2018.csv', delimiter=';')
    p2_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_DI2018.csv', delimiter=';')
    p3_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_DI2018.csv', delimiter=';')

    # uxux
    p1_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_DI2018.csv', delimiter=';')
    p1_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_KM2000.csv', delimiter=';')
    p1_R2016_uxux  = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_R2016.csv', delimiter=';')
    p2_BM1994_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_BM1994.csv', delimiter=';')
    p2_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_DI2018.csv', delimiter=';')
    p2_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_KM2000.csv', delimiter=';')
    p2_LS1993_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_LS1993.csv', delimiter=';')
    p2_R2016_uxux  = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_R2016.csv', delimiter=';')
    p3_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_DI2018.csv', delimiter=';')
    p3_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_KM2000.csv', delimiter=';')
    p3_R2016_uxux  = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_R2016.csv', delimiter=';')

    # uyuy
    p1_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos1_DI2018.csv', delimiter=';')
    p1_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos1_R2016.csv', delimiter=';')
    p2_BM1994_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_BM1994.csv', delimiter=';')
    p2_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_DI2018.csv', delimiter=';')
    p2_LS1993_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_LS1993.csv', delimiter=';')
    p2_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_R2016.csv', delimiter=';')
    p3_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos3_DI2018.csv', delimiter=';')
    p3_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos3_R2016.csv', delimiter=';')

    # uxuy
    p1_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_BM1994.csv', delimiter=';')
    p1_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_DI2018.csv', delimiter=';')
    p1_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_R2016.csv', delimiter=';')
    p2_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_BM1994.csv', delimiter=';')
    p2_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_DI2018.csv', delimiter=';')
    p2_LS1993_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_LS1993.csv', delimiter=';')
    p2_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_R2016.csv', delimiter=';')
    p3_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_BM1994.csv', delimiter=';')
    p3_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_DI2018.csv', delimiter=';')
    p3_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_R2016.csv', delimiter=';')

    # output sim data to files (not averaged over time)
    if output_data and output_velocity_profile:
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_1_timeseries.npy", np.array(AvgVelocity1.out))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_2_timeseries.npy", np.array(AvgVelocity2.out))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_3_timeseries.npy", np.array(AvgVelocity3.out))

    u1 = np.array(AvgVelocity1.out)[int(avg_u_start*np.array(AvgVelocity1.out).shape[0]-1):]
    u2 = np.array(AvgVelocity2.out)[int(avg_u_start*np.array(AvgVelocity2.out).shape[0]-1):]
    u3 = np.array(AvgVelocity3.out)[int(avg_u_start*np.array(AvgVelocity3.out).shape[0]-1):]

    avg_u1 = np.mean(u1, axis=0)  # time average
    avg_u2 = np.mean(u2, axis=0)  # time average
    avg_u3 = np.mean(u3, axis=0)  # time average

    if output_data:  # output (time-mean) velocity profiles
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_1_t-avg.npy", avg_u1)
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_2_t-avg.npy", avg_u2)
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_3_t-avg.npy", avg_u3)

    avg_u1_x = avg_u1[0]  # u_x component over y at pos 1
    avg_u2_x = avg_u2[0]  # u_x component over y at pos 2
    avg_u3_x = avg_u3[0]  # u_x component over y at pos 3

    avg_u1_y = avg_u1[1]  # u_y component over y at pos 1
    avg_u2_y = avg_u2[1]  # u_y component over y at pos 2
    avg_u3_y = avg_u3[1]  # u_y component over y at pos 3

    y_in_D = (np.arange(avg_u1_x.shape[0])+1-flow.y_pos)/flow.units.characteristic_length_lu  # y/D for figure
    if output_data:
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_YinD.npy", y_in_D)
    cm=1/2.54
    # PLOT ux
    fig, (ax_ux, ax_uy) = plt.subplots(1,2, constrained_layout=True, figsize=(30*cm,10*cm))
    ax_ux.plot(y_in_D,avg_u1_x, y_in_D, avg_u2_x, y_in_D, avg_u3_x)
    ax_ux.set_xlabel("y/D")
    ax_ux.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
    ax_ux.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

    # OPT. TO DO: add secondary axis for LU-grid
    # ...needs 'function' to convert from y/D in LU and LU in y/D

    # OPT. TO DO: make folder for AvgVelocity-stuff
    # if output_data:
    #     plt.savefig(output_path+dir_name+"/AvgVelocity_x.png")
    # plt.close()

    # PLOT uy
    #fig, ax = plt.subplots(constrained_layout=True)
    ax_uy.plot(y_in_D,avg_u1_y, y_in_D, avg_u2_y, y_in_D, avg_u3_y)
    ax_uy.set_xlabel("y/D")
    ax_uy.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
    ax_uy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

    # OPT. TO DO: add secondary axis for LU-grid
    # ...needs 'function' to convert from y/D in LU and LU in y/D
    # OPT. TO DO: make folder for AvgVelocity-stuff
    # !!! QUESTION: is x/D the position measured FROM the cylinder (x_pos), or measured from x=0 ?

    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_velocity_noReference.png")
    plt.close()

    # PLOT ux against references
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D,avg_u1_x, y_in_D, avg_u2_x-1, y_in_D, avg_u3_x-2)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_LS = ax.plot(p1_LS1993_ux[:,0], p1_LS1993_ux[:,1], p2_LS1993_ux[:,0], p2_LS1993_ux[:,1], p3_LS1993_ux[:,0], p3_LS1993_ux[:,1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_KM = ax.plot(p1_KM2000_ux[:,0], p1_KM2000_ux[:,1], p2_KM2000_ux[:,0], p2_KM2000_ux[:,1], p3_KM2000_ux[:,0], p3_KM2000_ux[:,1])
    plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k", label="Kravchenko & Moin (2000)")
    ref_WR = ax.plot(p1_WR2008_ux[:,0], p1_WR2008_ux[:,1], p2_WR2008_ux[:,0], p2_WR2008_ux[:,1], p3_WR2008_ux[:,0], p3_WR2008_ux[:,1])
    plt.setp(ref_WR, ls="dashdot", lw=1.5, marker="", color="k", label="Wissink & Rodi (2008)")
    ref_DI = ax.plot(p1_DI2018_ux[:,0], p1_DI2018_ux[:,1], p2_DI2018_ux[:,0], p2_DI2018_ux[:,1], p3_DI2018_ux[:,0], p3_DI2018_ux[:,1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
    ax.set_ylim([-2.5,+2])
    ax.set_xlim([-3,3])
    ax.legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0], ref_DI[0]], loc= 'best')
    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_ux_withReference.png")
    plt.close()

    # PLOT uy against references
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D,avg_u1_y, y_in_D, avg_u2_y-1, y_in_D, avg_u3_y-2)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_LS = ax.plot(p1_LS1993_uy[:,0], p1_LS1993_uy[:,1], p2_LS1993_uy[:,0], p2_LS1993_uy[:,1], p3_LS1993_uy[:,0], p3_LS1993_uy[:,1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_KM = ax.plot(p1_KM2000_uy[:,0], p1_KM2000_uy[:,1], p2_KM2000_uy[:,0], p2_KM2000_uy[:,1], p3_KM2000_uy[:,0], p3_KM2000_uy[:,1])
    plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k", label="Kravchenko & Moin (2000)")
    ref_WR = ax.plot(p1_WR2008_uy[:,0], p1_WR2008_uy[:,1], p2_WR2008_uy[:,0], p2_WR2008_uy[:,1], p3_WR2008_uy[:,0], p3_WR2008_uy[:,1])
    plt.setp(ref_WR, ls="dashdot", lw=1.5, marker="", color="k", label="Wissink & Rodi (2008)")
    ref_DI = ax.plot(p1_DI2018_uy[:,0], p1_DI2018_uy[:,1], p2_DI2018_uy[:,0], p2_DI2018_uy[:,1], p3_DI2018_uy[:,0], p3_DI2018_uy[:,1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
    ax.set_ylim([-2.5,+1.5])
    ax.set_xlim([-3,3])
    ax.legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0], ref_DI[0]], loc= 'best')
    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_uy_withReference.png")
    plt.close()

    ## turbulent Reynolds stresses

    # diff between timeseries and time_average -> u'
    u1_diff = u1-avg_u1
    u2_diff = u2-avg_u2
    u3_diff = u3-avg_u3

    # square of diff -> u'^2
    u1_diff_sq = u1_diff**2
    u2_diff_sq = u2_diff**2
    u3_diff_sq = u3_diff**2

    # ux'*uy'
    u1_diff_xy = u1_diff[:, 0, :]*u1_diff[:, 1, :]
    u2_diff_xy = u2_diff[:, 0, :]*u2_diff[:, 1, :]
    u3_diff_xy = u3_diff[:, 0, :]*u3_diff[:, 1, :]

    # time_average of u'² and ux'uy'
    u1_diff_sq_mean = np.mean(u1_diff_sq, axis=0)  # time average
    u2_diff_sq_mean = np.mean(u2_diff_sq, axis=0)  # time average
    u3_diff_sq_mean = np.mean(u3_diff_sq, axis=0)  # time average
    u1_diff_xy_mean = np.mean(u1_diff_xy, axis=0)  # time average
    u2_diff_xy_mean = np.mean(u2_diff_xy, axis=0)  # time average
    u3_diff_xy_mean = np.mean(u3_diff_xy, axis=0)  # time average

    if output_data:  # save reynolds stresses
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_1_ReStress_x.npy", np.array([y_in_D, u1_diff_sq_mean[0]]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_2_ReStress_x.npy", np.array([y_in_D, u2_diff_sq_mean[0]]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_3_ReStress_x.npy", np.array([y_in_D, u3_diff_sq_mean[0]]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_1_ReStress_y.npy", np.array([y_in_D, u1_diff_sq_mean[1]]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_2_ReStress_y.npy", np.array([y_in_D, u2_diff_sq_mean[1]]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_3_ReStress_y.npy", np.array([y_in_D, u3_diff_sq_mean[1]]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_1_ReShearStress.npy", np.array([y_in_D, u1_diff_xy_mean]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_2_ReShearStress.npy", np.array([y_in_D, u2_diff_xy_mean]))
        np.save(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_3_ReShearStress.npy", np.array([y_in_D, u3_diff_xy_mean]))


    fig, (ax_xx, ax_yy, ax_xy) = plt.subplots(1,3, figsize=(40*cm,10*cm), constrained_layout=True)
    ax_xx.plot(y_in_D,u1_diff_sq_mean[0],y_in_D,u2_diff_sq_mean[0],y_in_D,u3_diff_sq_mean[0])
    ax_xx.set_xlabel("y/D")
    ax_xx.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
    ax_xx.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    # if output_data:
    #     plt.savefig(output_path+dir_name+"/AvgVelocity_uxux.png")
    # plt.close()

    #fig, ax = plt.subplots(constrained_layout=True)
    ax_yy.plot(y_in_D,u1_diff_sq_mean[1], y_in_D,u2_diff_sq_mean[1], y_in_D,u3_diff_sq_mean[1])
    ax_yy.set_xlabel("y/D")
    ax_yy.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
    ax_yy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    # if output_data:
    #     plt.savefig(output_path+dir_name+"/AvgVelocity_uyuy.png")
    # plt.close()

    #fig, ax = plt.subplots(constrained_layout=True)
    ax_xy.plot(y_in_D,u1_diff_xy_mean, y_in_D,u2_diff_xy_mean, y_in_D,u3_diff_xy_mean)
    ax_xy.set_xlabel("y/D")
    ax_xy.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
    ax_xy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_reynoldsStresses_noReference.png")
    plt.close()

    #plot reynolds stresses against reference
    # uxux - streamwise
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D,u1_diff_sq_mean[0], y_in_D, u2_diff_sq_mean[0]-0.5, y_in_D, u3_diff_sq_mean[0]-1)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_LS = ax.plot(p2_LS1993_uxux[:,0], p2_LS1993_uxux[:,1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_R = ax.plot(p1_R2016_uxux[:,0], p1_R2016_uxux[:,1], p3_R2016_uxux[:,0], p3_R2016_uxux[:,1], p3_R2016_uxux[:,0], p3_R2016_uxux[:,1])
    plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k", label="Rajani et al. (2016)")
    ref_KM = ax.plot(p1_KM2000_uxux[:,0], p1_KM2000_uxux[:,1], p2_KM2000_uxux[:,0], p2_KM2000_uxux[:,1], p3_KM2000_uxux[:,0], p3_KM2000_uxux[:,1])
    plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k", label="Kravchenko & Moin (2000)")
    ref_BM = ax.plot(p2_BM1994_uxux[:,0], p2_BM1994_uxux[:,1])
    plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k", label="Beaudan & Moin (1994)")
    ref_DI = ax.plot(p1_DI2018_uxux[:,0], p1_DI2018_uxux[:,1], p2_DI2018_uxux[:,0], p2_DI2018_uxux[:,1], p3_DI2018_uxux[:,0], p3_DI2018_uxux[:,1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
    ax.set_ylim([-1.2,0.8])
    ax.set_xlim([-3,3])
    ax.legend(handles=[my_data[0], ref_LS[0], ref_R[0], ref_KM[0], ref_BM[0], ref_DI[0]], loc= 'best')

    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_uxux_withReference.png")
    plt.close()

    # uyuy - cross-stream
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D,u1_diff_sq_mean[1], y_in_D, u2_diff_sq_mean[1]-0.5, y_in_D, u3_diff_sq_mean[1]-1)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_BM = ax.plot(p2_BM1994_uyuy[:,0], p2_BM1994_uyuy[:,1])
    plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k", label="Beaudan & Moin (1994)")
    ref_LS = ax.plot(p2_LS1993_uyuy[:,0], p2_LS1993_uyuy[:,1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_R = ax.plot(p1_R2016_uyuy[:,0], p1_R2016_uyuy[:,1], p3_R2016_uyuy[:,0], p3_R2016_uyuy[:,1], p3_R2016_uyuy[:,0], p3_R2016_uyuy[:,1])
    plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k", label="Rajani et al. (2016)")
    ref_DI = ax.plot(p1_DI2018_uyuy[:,0], p1_DI2018_uyuy[:,1], p2_DI2018_uyuy[:,0], p2_DI2018_uyuy[:,1], p3_DI2018_uyuy[:,0], p3_DI2018_uyuy[:,1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
    ax.set_ylim([-1.2,0.8])
    ax.set_xlim([-3,3])
    ax.legend(handles=[my_data[0], ref_BM[0], ref_LS[0], ref_R[0], ref_DI[0]], loc='best')

    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_uyuy_withReference.png")
    plt.close()

    # uxuy - Reynolds shear stress
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D,u1_diff_xy_mean, y_in_D, u2_diff_xy_mean-0.5, y_in_D, u3_diff_xy_mean-1)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_BM = ax.plot(p2_BM1994_uxuy[:,0], p2_BM1994_uxuy[:,1])
    plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k", label="Beaudan & Moin (1994)")
    ref_LS = ax.plot(p2_LS1993_uxuy[:,0], p2_LS1993_uxuy[:,1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_R = ax.plot(p1_R2016_uxuy[:,0], p1_R2016_uxuy[:,1], p3_R2016_uxuy[:,0], p3_R2016_uxuy[:,1], p3_R2016_uxuy[:,0], p3_R2016_uxuy[:,1])
    plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k", label="Rajani et al. (2016)")
    ref_DI = ax.plot(p1_DI2018_uxuy[:,0], p1_DI2018_uxuy[:,1], p2_DI2018_uxuy[:,0], p2_DI2018_uxuy[:,1], p3_DI2018_uxuy[:,0], p3_DI2018_uxuy[:,1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
    ax.set_ylim([-1.2,0.8])
    ax.set_xlim([-3,3])
    ax.legend(handles=[my_data[0], ref_BM[0], ref_LS[0], ref_R[0], ref_DI[0]], loc='best')

    if output_data:
        plt.savefig(output_path+dir_name+"/AvgVelocity_Data"+"/AvgVelocity_uxuy_withReference.png")
    plt.close()

    # (!) standard plot/figure size in python is 6.4 x 4.8 inches

# DRAG COEFFICIENT
drag_coefficient = np.array(Dragreport.out)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(drag_coefficient[:,1],drag_coefficient[:,2])
ax.set_xlabel("physical time / s")
ax.set_ylabel("Coefficient of Drag Cd")
ax.set_ylim([0.5,1.6])
secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
if output_data:
    plt.savefig(output_path+dir_name+"/drag_coefficient.png")
    np.savetxt(output_path+dir_name+"/drag_coefficient.txt", drag_coefficient, header="stepLU  |  timePU  |  Cd  FROM str(timestamp)")
ax.set_ylim([drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].min()*0.5,drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].max()*1.2])
if output_data:
    plt.savefig(output_path+dir_name+"/drag_coefficient_adjusted.png")
plt.close()

# peak finder: try calculating the mean drag coefficient from an integer number of periods, if a clear periodic signal is found
try:
    values = drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2]

    peaks_max = find_peaks(values, prominence=((values.max()-values.min())/2))
    peaks_min = find_peaks(-values, prominence=((values.max()-values.min())/2))
    if peaks_min[0].shape[0] - peaks_max[0].shape[0] > 0:
        peak_number = peaks_max[0].shape[0]
    else:
        peak_number = peaks_min[0].shape[0]

    if peaks_min[0][0] < peaks_max[0][0]:
        first_peak = peaks_min[0][0]
        last_peak = peaks_max[0][peak_number-1]
    else:
        first_peak = peaks_max[0][0]
        last_peak = peaks_min[0][peak_number-1]

    drag_mean = values[first_peak:last_peak].mean()
    drag_mean_simple = values.mean()

    print("Cd, simple mean:     ",drag_mean_simple)
    print("Cd, peak_finder mean:",drag_mean)

    drag_stepsLU = drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,0]
    peak_max_y = values[peaks_max[0]]
    peak_max_x = drag_stepsLU[peaks_max[0]]
    peak_min_y = values[peaks_min[0]]
    peak_min_x = drag_stepsLU[peaks_min[0]]

    plt.plot(drag_stepsLU, values)
    plt.scatter(peak_max_x[:peak_number],peak_max_y[:peak_number])
    plt.scatter(peak_min_x[:peak_number],peak_min_y[:peak_number])
    plt.scatter(drag_stepsLU[first_peak],values[first_peak])
    plt.scatter(drag_stepsLU[last_peak],values[last_peak])
    plt.savefig(output_path+dir_name+"/drag_coefficient_peakfinder.png")
    peakfinder=True
except: # if signal is not sinusoidal enough, calculate only simple mean value
    print("peak-finding didn't work... probably no significant peaks visible (Re<46?), or periodic region not reached (T too small)")
    values = drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2]
    drag_mean_simple = values.mean()
    peakfinder=False
    print("Cd, simple mean:",drag_mean_simple)
plt.close()

# LIFT COEFFICIENT
lift_coefficient = np.array(Liftreport.out)
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(lift_coefficient[:,1],lift_coefficient[:,2])
ax.set_xlabel("physical time / s")
ax.set_ylabel("Coefficient of Lift Cl")
ax.set_ylim([-1.1,1.1])

secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")
if output_data:
    plt.savefig(output_path+dir_name+"/lift_coefficient.png")
    np.savetxt(output_path+dir_name+"/lift_coefficient.txt", lift_coefficient, header="stepLU  |  timePU  |  Cl  FROM str(timestamp)")
Cl_min = lift_coefficient[int(lift_coefficient[:,2].shape[0]*0.5):,2].min()
Cl_max = lift_coefficient[int(lift_coefficient[:,2].shape[0]*0.5):,2].max()
print("Cl_peaks: \nmin", Cl_min,"\nmax", Cl_max)
plt.close()

# plot DRAG and LIFT together:
fig, ax = plt.subplots(layout="constrained")
drag_ax = ax.plot(drag_coefficient[:,1],drag_coefficient[:,2], color="tab:blue", label="Drag")
ax.set_xlabel("physical time / s")
ax.set_ylabel("Coefficient of Drag Cd")
ax.set_ylim([0.5,1.6])

secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
secax.set_xlabel("timesteps (simulation time / LU)")

ax2 = ax.twinx()
lift_ax = ax2.plot(lift_coefficient[:,1],lift_coefficient[:,2], color="tab:orange", label="Lift")
ax2.set_ylabel("Coefficient of Lift Cl")
ax2.set_ylim([-1.1,1.1])


fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

if output_data:
    plt.savefig(output_path+dir_name+"/dragAndLift_coefficient.png")
plt.close()

# STROUHAL number: (only makes sense for Re>46 and if periodic state is reached)
try:
    ### prototyped fft for frequency detection and calculation of strouhal-number
    # ! Drag_frequency is 2* Strouhal-Freq. Lift-freq. is Strouhal-Freq.

    X = np.fft.fft(lift_coefficient[:,2])   # fft result (amplitudes)
    N = len(X)  # number of freqs
    n = np.arange(N)   # freq index
    T = N*flow.units.convert_time_to_pu(1)   # total time measured (T_PU)
    freq = n/T   # frequencies (x-axis of spectrum)

    plt.figure
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")   # plot spectrum |X|(f)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.xlim(0,1)
    #print("max. Amplitude np.abx(X).max():", np.abs(X).max())   # for debugging
    plt.ylim(0,np.abs(X[:int(X.shape[0]*0.5)]).max())   # ylim, where highes peak is on left half of full spectrum

    if output_data:
        plt.savefig(output_path+dir_name+"/fft_Cl.png")

    freq_res = freq[1]-freq[0]   # frequency-resolution
    X_abs = np.abs(X[:int(X.shape[0]*0.4)])   # get |X| Amplitude for left half of full spectrum
    freq_peak = freq[np.argmax(X_abs)]    # find frequency with highest amplitude
    print("Frequency Peak:", freq_peak, "+-", freq_res, "Hz")
    # f = Strouhal for St=f*D/U and D=U=1 in PU
except:
    print("fft for Strouhal didn't work")
    freq_res = 0
    freq_peak = 0
plt.close()

##################################################
# OUTPUT DATA and stats to directory

### export VRAM info:

if output_data:
### CUDA-VRAM-summary:
    output_file = open(output_path+dir_name+"/"+timestamp + "_GPU_memory_summary.txt", "a")
    output_file.write("DATA for "+timestamp+"\n\n")
    output_file.write(torch.cuda.memory_summary(lattice.device))
    output_file.close()
    
### list present torch tensors:
    output_file = open(output_path+dir_name+"/"+timestamp + "_GPU_list_of_tensors.txt", "a")
    total_bytes = 0
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj,'data') and torch.is_tensor(obj.data)):
                output_file.write("\n"+str(obj.size())+", "+str(obj.nelement()*obj.element_size()))
                total_bytes = total_bytes+obj.nelement()*obj.element_size()    
        except:
            pass
    #output_file.write("\n\ntotal bytes for tensors:"+str(total_bytes))
    output_file.close()

### count occurence of tensors in list of tensors:
    from collections import Counter
    my_file = open(output_path+dir_name+"/"+timestamp + "_GPU_list_of_tensors.txt","r")
    data=my_file.read()
    my_file.close()
    data_into_list=data.split("\n")
    c = Counter(data_into_list)
    output_file = open(output_path+dir_name+"/"+timestamp + "_GPU_counted_tensors.txt", "a")
    for k,v in c.items():
        output_file.write("type,size,bytes: {}, number: {}\n".format(k,v) )
    output_file.write("\ntotal bytes for tensors:"+str(total_bytes))
    output_file.close()

# output parameters, stats and observables
if output_data:
    output_file = open(output_path+dir_name+"/"+timestamp + "_parms_stats_obs.txt", "a")
    output_file.write("DATA for "+timestamp)
    output_file.write("\n\n###   SIM-Parameters   ###")
    output_file.write("\nRe = "+str(re))
    output_file.write("\nn_steps = "+str(n_steps))
    output_file.write("\nT_target = "+str(flow.units.convert_time_to_pu(n_steps))+" seconds")
    output_file.write("\ngridpoints_per_diameter (gpd) = "+str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: "+str(gpd_setup)+" to "+str(gridpoints_per_diameter)+" because D/Y is even")
    output_file.write("\nDpX (D/X) = " + str(domain_length_in_D))
    output_file.write("\nDpY (D/Y) = "+str(domain_height_in_D))
    if lattice.D == 3:
        output_file.write("\nDpZ (D/Z) = "+str(domain_width_in_D))
    output_file.write("\nshape_LU: "+ str(flow.shape))
    output_file.write(("\ntotal_number_of_gridpoints: "+str(lattice.rho(sim.f).numel())))
    output_file.write("\nbc_type = "+str(bc_type))
    output_file.write("\nlateral_walls = "+str(lateral_walls))
    output_file.write("\nstencil = "+str(stencil_choice))
    output_file.write("\ncollision = " + str(collision_choice))
    output_file.write("\n")
    output_file.write("\nMa = " + str(Ma))
    output_file.write("\ntau = " + str(tau))
    output_file.write("\ngrid_reynolds_number (Re_g) = " + str(re_g))
    output_file.write("\n")
    output_file.write("\nsetup_diameter_PU = " + str(setup_diameter))
    output_file.write("\nflow_velocity_PU = " + str(flow_velocity))
    output_file.write("\nu_init = " + str(u_init))
    output_file.write("\nperturb_init = " + str(perturb_init))
    output_file.write("\n")
    output_file.write("\noutput_vtk = " + str(output_vtk))
    output_file.write("\nvtk_fps = " + str(vtk_fps))

    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = "+str(runtime)+ " seconds (="+str(runtime/60)+" minutes)")
    output_file.write("\nMLUPS = "+str(mlups))
    output_file.write("\n")
    output_file.write("\navg. Collision-Time [s] = " + str(c_time) + " (" + str(round(100 * c_time/sum_time, 2)) + " %)")
    output_file.write("\navg. store_fcl-Time [s] = " + str(fc_time) + " (" + str(round(100 * fc_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Streaming-Time [s] = " + str(s_time) + " (" + str(round(100 * s_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Boundary-Time  [s] = " + str(b_time) + " (" + str(round(100 * b_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Reporter-Time  [s] = " + str(r_time) + " (" + str(round(100 * r_time/sum_time, 2)) + " %)")
    output_file.write("\n")
    output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(lattice.device)/1024/1024))
    output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(lattice.device)/1024/1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\ntotal current RAM usage [MB]: " + str(round(ram.used/(1024*1024),2)) + " of " + str(round(ram.total/(1024*1024),2)) + " MB")

    output_file.write("\n\n###   OBSERVABLES   ###")
    output_file.write("\nCoefficient of drag between "+str(round(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1),1],2))+" s and "+str(round(drag_coefficient[int(drag_coefficient.shape[0]-1),1],2))+" s:")
    output_file.write("\nCd_mean, simple      = "+str(drag_mean_simple))
    if peakfinder:
        output_file.write("\nCd_mean, peak_finder = "+str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write("\nCd_min = "+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].min()))
    output_file.write("\nCd_max = "+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].max()))
    output_file.write("\n")
    output_file.write("\nCoefficient of lift:")
    output_file.write("\nCl_min = "+str(Cl_min))
    output_file.write("\nCl_max = "+str(Cl_max))
    output_file.write("\n")
    output_file.write("\nStrouhal number:")
    output_file.write("\nSt +- df = "+str(freq_peak)+" +- "+str(freq_res)+" Hz")
    output_file.write("\n")
    output_file.close()

#output copyable numbers for EXCEL etc.
if output_data:
    output_file = open(output_path+dir_name+"/"+timestamp + "_parms_stats_obs_copyable.txt", "a")

    output_file.write("DATA for "+timestamp)
    output_file.write("\n\n###   Data:   ###")
    output_file.write("\nRe, n_steps, t_target(PU), GPD, DpX, DpY, (DpZ), shape_LU, gridpoints, bc_type, lateral_walls, stencil, collision, Ma, tau, Re_grid, setup_diameter_PU, flow_velocity_PU, u_init, perturb_init, output_vtk, vtk_fps, runtime, MLUPS, c_time, fc_time, s_time, b_time, r_time, VRAM_current_MB, VRAM_peak_MB, periodic_start, Cd_mean, Cd_mean_pf, Cd_min, Cd_max, Cl_min, Cl_max, St, df\n")
    output_file.write("\n" + str(re))
    output_file.write("\n" + str(n_steps))
    output_file.write("\n" + str(flow.units.convert_time_to_pu(n_steps)))
    output_file.write("\n" + str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: "+str(gpd_setup)+" to "+str(gridpoints_per_diameter)+" because D/Y is even")
    output_file.write("\n" + str(domain_length_in_D))
    output_file.write("\n" + str(domain_height_in_D))
    if lattice.D == 3:
        output_file.write("\n" + str(domain_width_in_D))
    output_file.write("\n" + str(flow.shape))
    output_file.write("\n" + str(lattice.rho(sim.f).numel()))
    output_file.write("\n"+str(bc_type))
    output_file.write("\n"+str(lateral_walls))
    output_file.write("\n"+str(stencil_choice))
    output_file.write("\n" + str(collision_choice))
    output_file.write("\n")
    output_file.write("\n" + str(Ma))
    output_file.write("\n" + str(tau))
    output_file.write("\n" + str(re_g))
    output_file.write("\n")
    output_file.write("\n" + str(setup_diameter))
    output_file.write("\n" + str(flow_velocity))
    output_file.write("\n" + str(u_init))
    output_file.write("\n" + str(perturb_init))
    output_file.write("\n")
    output_file.write("\n" + str(output_vtk))
    output_file.write("\n" + str(vtk_fps))
    output_file.write("\n")

    output_file.write("\n"+str(runtime))
    output_file.write("\n"+str(mlups))
    output_file.write("\n")
    output_file.write("\n" + str(c_time))
    output_file.write("\n" + str(fc_time))
    output_file.write("\n" + str(s_time))
    output_file.write("\n" + str(b_time))
    output_file.write("\n" + str(r_time))
    output_file.write("\n")
    output_file.write("\n" + str(torch.cuda.memory_allocated(lattice.device)/1024/1024))
    output_file.write("\n" + str(torch.cuda.max_memory_allocated(lattice.device)/1024/1024))
    output_file.write("\n")

    output_file.write("\n" + str(periodic_start))
    output_file.write("\n"+str(drag_mean_simple))
    if peakfinder:
        output_file.write("\n"+str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write("\n"+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].min()))
    output_file.write("\n"+str(drag_coefficient[int(drag_coefficient.shape[0]*periodic_start-1):,2].max()))
    output_file.write("\n")
    output_file.write("\n"+str(Cl_min))
    output_file.write("\n"+str(Cl_max))
    output_file.write("\n")
    output_file.write("\n"+str(freq_peak))
    output_file.write("\n"+str(freq_res))
    output_file.write("\n")
    output_file.close()

