import lettuce as lt
import numpy as np
import torch

"""
Setup simulation
-----
"""
# lattice = lt.Lattice(lt.D3Q27, 'cuda')
# streaming = lt.StandardStreaming(lattice)
# flow = lt.TaylorGreenVortex3D(resolution=120, reynolds_number=1600, mach_number=0.05, lattice=lattice)
# collision = lt.SmagorinskyCollision(lattice, tau=flow.units.relaxation_parameter_lu)
# simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
# simulation.reporters.append(lt.EnstrophyReporter(lattice, flow, interval=10, out=None))
# simulation.reporters.append(lt.EnergyReporter(lattice, flow, interval=10, out=None))
# simulation.reporters.append(lt.EnergyReporter(lattice, flow, interval=500))
# simulation.step(num_steps=4000)
# path = "/home/mario/Dokumente/science/plots/NeuronalNetwork/"
# np.savetxt(path+"tgv3d_smag_Re1600_Res120_enstrophy.csv", simulation.reporters[0].out, delimiter=';')
# np.savetxt(path+"tgv3d_smag_Re1600_Res120_energy.csv", simulation.reporters[1].out, delimiter=';')




lattice = lt.Lattice(lt.D3Q27, 'cuda')
streaming = lt.StandardStreaming(lattice)
flow = lt.TaylorGreenVortex3D(resolution=60, reynolds_number=1600, mach_number=0.05, lattice=lattice)
net = torch.load('neuronal_network_data.pt')
collision = lt.SmagorinskyCollision_nn(lattice, tau=flow.units.relaxation_parameter_lu,net=net)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
simulation.reporters.append(lt.EnstrophyReporter(lattice, flow, interval=10, out=None))
simulation.reporters.append(lt.EnergyReporter(lattice, flow, interval=10, out=None))
simulation.reporters.append(lt.EnergyReporter(lattice, flow, interval=500))

simulation.step(num_steps=8000)
path = "/home/mario/Dokumente/science/plots/NeuronalNetwork/"
np.savetxt(path+"tgv3d_smag_nn_Re1600_Res120_enstrophy.csv", simulation.reporters[0].out, delimiter=';')
np.savetxt(path+"tgv3d_smag_nn_Re1600_Res120_energy.csv", simulation.reporters[1].out, delimiter=';')

print(flow.units.viscosity_pu)
print((flow.units.convert_time_to_pu(1)))