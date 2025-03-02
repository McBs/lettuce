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
parser.add_argument("--Precision", type=str, choices=["Single", "Double", "Half"], default="Double")
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
Precision = args.Precision

# Falls Ordner nicht existieren, erstelle sie
os.makedirs(csvdir, exist_ok=True)
os.makedirs(vtkdir, exist_ok=True)

# Setze die Präzision
if Precision == "Single":
    dtype = torch.float32
elif Precision == "Double":
    dtype = torch.float64
elif Precision == "Half":
    dtype = torch.float16

# Lattice und Flow Setup
lattice = lt.Lattice(lt.D3Q27, device="cuda", dtype=dtype)
flow = lt.TaylorGreenVortex3D(resolution=res, reynolds_number=Re, mach_number=Mach, lattice=lattice)

# Auswahl des Kollisionsmodells
if collision_operator == "BGK":
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
elif collision_operator == "KBC":
    collision = lt.KBCCollision3D(lattice, tau=flow.units.relaxation_parameter_lu)

# Simulation initialisieren
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)

# Reporter hinzufügen
interval_out = int((flow.units.convert_time_to_lu(tmax)) / 500)
interval_vtk = int((flow.units.convert_time_to_lu(tmax)) / 20)

simulation.reporters.append(lt.ObservableReporter(lt.IncompressibleKineticEnergy(lattice, flow), interval=interval_out, out=None))
simulation.reporters.append(lt.ObservableReporter(lt.Dissipation_sij(lattice, flow), interval=interval_out, out=None))
simulation.reporters.append(lt.ObservableReporter(lt.Enstrophy(lattice, flow), interval=interval_out, out=None))
simulation.reporters.append(lt.ObservableReporter(lt.EnergySpectrum2(lattice, flow), interval=interval_out, out=None))
simulation.reporters.append(lt.ObservableReporter(lt.SymmetryReporter(lattice, flow), interval=interval_out, out=None))
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=interval_vtk, filename_base=vtkdir + "out"))

# Simulation starten
simulation.initialize_f_neq()
steps = int(flow.units.convert_time_to_lu(tmax))
mlups = simulation.step(num_steps=steps)
print("Performance in MLUPS:", mlups)

# Daten aus den Reportern extrahieren
energy = np.array(simulation.reporters[0].out)
dissipation = np.array(simulation.reporters[1].out)
enstrophy = np.array(simulation.reporters[2].out)
energy_spectrum = np.array(simulation.reporters[3].out)
symmetry = np.array(simulation.reporters[4].out)

# Dissipation berechnen
dt = energy[1, 1] - energy[0, 1]
dEdt = np.gradient(energy[:, 2], dt)
dissipation_ekin = np.array([energy[:, 1], -dEdt / (np.pi * 2) ** 3])
dissipation_ens = np.array([enstrophy[:, 1], flow.units.viscosity_pu / (np.pi * 2) ** 3 * enstrophy[:, 2]])
dissipation_sij = np.array([dissipation[:, 1], dissipation[:, 2]])

# Daten speichern
np.savetxt(csvdir + "DissipationEkin.csv", dissipation_ekin.T, delimiter=",")
np.savetxt(csvdir + "DissipationEnstrophy.csv", dissipation_ens.T, delimiter=",")
np.savetxt(csvdir + "DissipationSij.csv", dissipation_sij.T, delimiter=",")
np.savetxt(csvdir + "EnergySpectrum.csv", energy_spectrum, delimiter=",")
np.savetxt(csvdir + "Symmetry.csv", symmetry, delimiter=",")

# Referenzdaten laden
if Re in [100, 800]:
    ref_file = f"/home/ben/Schreibtisch/Enstropy Brachet/Brachet/Brachet_1983_tgv3d_enstrophy_Re{Re}.csv"
    ref_data = np.loadtxt(ref_file, delimiter=";", skiprows=1).T
elif Re == 1600:
    ref_file = f"/home/ben/Schreibtisch/Referenz/Re{Re}.csv"
    ref_data = np.genfromtxt(ref_file, delimiter=";", dtype=str)
    ref_data = np.char.replace(ref_data, ',', '.').astype(float).T
else:
    ref_file = f"/home/ben/Schreibtisch/Referenz/Re{Re}.csv"
    ref_data = np.loadtxt(ref_file, delimiter=",")

# Plot: Vergleich mit Referenz
plt.figure()
plt.plot(ref_data[0, :], ref_data[1, :], label="Referenz", linestyle="dashed")
plt.plot(dissipation_ekin[0, :], dissipation_ekin[1, :], label="Standard", marker="o")
plt.xlabel("Zeit")
plt.ylabel("Dissipation")
plt.legend()
plt.title(f"Dissipation Vergleich Re={Re}")
plt.savefig(csvdir + "Dissipation_Comparison.png")
plt.show()

# Plot: Relative Differenz
plt.figure()
plt.plot(dissipation_ekin[0, :], (dissipation_ekin[1, :] - dissipation_sij[1, :]) / dissipation_ekin[1, :], label="Rel. Diff")
plt.xlabel("Zeit")
plt.ylabel("Relative Differenz")
plt.legend()
plt.title("Relative Dissipation Differenz")
plt.savefig(csvdir + "Dissipation_Relative_Diff.png")
plt.show()
