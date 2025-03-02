import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import csv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Argument Parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--vtkdir", type=str, default="./output/", help="Directory for VTK files")
parser.add_argument("--csvdir", type=str, default="./output/", help="Directory for CSV output")
parser.add_argument("--resolution", type=int, default=32)
parser.add_argument("--nout", type=int, default=100)
parser.add_argument("--nvtk", type=int, default=200)
parser.add_argument("--tmax", type=int, default=20)
parser.add_argument("--Re", type=int, default=100)
parser.add_argument("--collision_operator", type=str, choices=["BGK", "KBC"], default="BGK")
parser.add_argument("--Mach", type=float, default=0.005)

args = parser.parse_args()

# Parameter setzen
Re = args.Re
csvdir = args.csvdir
vtkdir = args.vtkdir
res = args.resolution
tmax = args.tmax
collision_operator = args.collision_operator
Mach = args.Mach


# Falls Ordner nicht existieren, erstelle sie
os.makedirs(csvdir, exist_ok=True)
os.makedirs(vtkdir, exist_ok=True)

def compute_dissipation(simulation):
    energy = np.array(simulation.reporters[0].out)
    dEdt = np.zeros(len(energy[:, 2]))
    dt = energy[1, 1] - energy[0, 1]
    nmax = len(energy[:, 2])

    for i in range(nmax):
        if i == 0:
            dEdt[i] = (energy[1, 2] - energy[0, 2]) / dt
        elif i == 1:
            dEdt[i] = (energy[2, 2] - energy[0, 2]) / (2 * dt)
        elif i == 2:
            dEdt[i] = (1 / 12 * energy[0, 2] - 2 / 3 * energy[1, 2] + 2 / 3 * energy[3, 2] - 1 / 12 * energy[4, 2]) / dt
        elif i >= (nmax - 3):
            if i == (nmax - 3):
                dEdt[i] = (1 / 12 * energy[nmax - 5, 2] - 2 / 3 * energy[nmax - 4, 2] + 2 / 3 * energy[
                    nmax - 2, 2] - 1 / 12 * energy[nmax - 1, 2]) / dt
            elif i == (nmax - 2):
                dEdt[i] = (energy[nmax - 1, 2] - energy[nmax - 3, 2]) / (2 * dt)
            elif i == (nmax - 1):
                dEdt[i] = (energy[nmax - 1, 2] - energy[nmax - 2, 2]) / dt
        else:
            dEdt[i] = (-1 / 60 * energy[i - 3, 2] + 3 / 20 * energy[i - 2, 2] - 3 / 4 * energy[i - 1, 2] +
                       3 / 4 * energy[i + 1, 2] - 3 / 20 * energy[i + 2, 2] + 1 / 60 * energy[i + 3, 2]) / dt

    return np.array([energy[:, 1], -dEdt / (np.pi * 2) ** 3])


# **1️⃣ Simulation mit Float64 für f & tau**
lattice_64 = lt.Lattice(lt.D3Q27, device="cuda", dtype=torch.float64)
flow_64 = lt.TaylorGreenVortex3D(resolution=res, reynolds_number=Re, mach_number=Mach, lattice=lattice_64)
collision_64_tau64 = lt.BGKCollision(lattice_64, tau=flow_64.units.relaxation_parameter_lu)
simulation_64_tau64 = lt.Simulation(flow=flow_64, lattice=lattice_64, collision=collision_64_tau64,
                                    streaming=lt.StandardStreaming(lattice_64))
inter = int((flow_64.units.convert_time_to_lu(tmax)) / 500)


simulation_64_tau64.reporters.append(
    lt.ObservableReporter(lt.IncompressibleKineticEnergy(lattice_64, flow_64), interval=inter, out=None))
simulation_64_tau64.initialize_f_neq()
simulation_64_tau64.step(num_steps=int(flow_64.units.convert_time_to_lu(tmax)))
dissipation_64_tau64 = compute_dissipation(simulation_64_tau64)

# **2️⃣ Simulation mit Float64 für f, aber Tau in Float32**
collision_64_tau32 = lt.BGKCollision(lattice_64, tau=np.float64(np.float32(flow_64.units.relaxation_parameter_lu)))
simulation_64_tau32 = lt.Simulation(flow=flow_64, lattice=lattice_64, collision=collision_64_tau32,
                                    streaming=lt.StandardStreaming(lattice_64))
simulation_64_tau32.reporters.append(
    lt.ObservableReporter(lt.IncompressibleKineticEnergy(lattice_64, flow_64), interval=inter, out=None))
simulation_64_tau32.initialize_f_neq()
simulation_64_tau32.step(num_steps=int(flow_64.units.convert_time_to_lu(tmax)))
dissipation_64_tau32 = compute_dissipation(simulation_64_tau32)

# **3️⃣ Simulation mit Float32 Initialisierung, dann Float64**
lattice_32 = lt.Lattice(lt.D3Q27, device="cuda", dtype=torch.float32)
flow_32 = lt.TaylorGreenVortex3D(resolution=res, reynolds_number=Re, mach_number=Mach, lattice=lattice_32)
collision_32 = lt.BGKCollision(lattice_32, tau=flow_32.units.relaxation_parameter_lu)
simulation_32 = lt.Simulation(flow=flow_32, lattice=lattice_32, collision=collision_32,
                              streaming=lt.StandardStreaming(lattice_32))
simulation_32.initialize_f_neq()
f_64 = simulation_32.f.to(torch.float64)

# Neue Simulation mit Float64
simulation_64_init32 = lt.Simulation(flow=flow_64, lattice=lattice_64, collision=collision_64_tau64,
                                     streaming=lt.StandardStreaming(lattice_64))
simulation_64_init32.f = f_64
simulation_64_init32.reporters.append(
    lt.ObservableReporter(lt.IncompressibleKineticEnergy(lattice_64, flow_64), interval=inter, out=None))
simulation_64_init32.step(num_steps=int(flow_64.units.convert_time_to_lu(tmax)))
dissipation_init32_rest64 = compute_dissipation(simulation_64_init32)

# **Daten speichern**
np.savetxt(csvdir + "DissipationEkin_BGK64_tau64.csv", dissipation_64_tau64.T, delimiter=",")
np.savetxt(csvdir + "DissipationEkin_BGK64_tau32.csv", dissipation_64_tau32.T, delimiter=",")
np.savetxt(csvdir + "DissipationEkin_Init32_rest64.csv", dissipation_init32_rest64.T, delimiter=",")

# **Plots**
plt.figure()