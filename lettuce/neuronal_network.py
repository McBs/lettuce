import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LBMNet(nn.Module):
    def __init__(self):
        super(LBMNet, self).__init__()
        self.lin1 = nn.Linear(27, 300)
        self.lin2 = nn.Linear(300, 200)
        self.lin3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

device='cuda:0'
net = LBMNet()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=.01)
# optimizer = optim.Adam(net.parameters(), lr=.01)

#### Training

# if os.path.isfile('data_netz.pt'):
#     net = torch.load('data_netz.pt')

# import lettuce
# import lettuce as lt
#
# lattice = lt.Lattice(lt.D3Q27,device)
# flow = lt.TaylorGreenVortex3D(100,1600,0.1,lattice)
# collision = lt.SmagorinskyCollision(lattice,tau=flow.units.relaxation_parameter_lu)
# streaming = lt.StandardStreaming(lattice)
# simulation = lt.Simulation(flow,lattice,collision,streaming)
# simulation.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=1, out=None))
#
# for i in range(100):
#     simulation.step(num_steps=1)
#
#     optimizer.zero_grad()
#
#     input = simulation.collision.S_shear.permute(1,2,3,0)
#     target = simulation.collision.tau_eff[...,None]
#
#     out = net(input)
#     loss = criterion(out,target)
#     loss.backward()
#
#     optimizer.step()
#
#     if i % 50 == 0:
#         error = loss.item()
#         print(f"[{i + 1:4d}] running. Loss: {error / 1:.10f}")
#
#
# torch.save(net, 'neuronal_network_data.pt')
# #np.savetxt('tmp_training.csv', simulation.reporters[0].out, delimiter=';')
# print("Finish Training")

##### Test Net #######

import lettuce
import lettuce as lt
device='cuda:0'
lattice = lt.Lattice(lt.D3Q27,device)
flow = lt.TaylorGreenVortex3D(100,1600,0.1,lattice)
streaming = lt.StandardStreaming(lattice)

collision = lt.SmagorinskyCollision(lattice,tau=flow.units.relaxation_parameter_lu,net='neuronal_network_data.pt')
#collision = lt.BGKCollision(lattice,tau=flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow,lattice,collision,streaming)
simulation.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=1, out=None))
simulation.reporters.append(lt.EnstrophyReporter(lattice=lattice, flow=flow , interval=1, out=None))
simulation.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=100))
print("Convert 1 timestep in lu to pu: ", flow.units.convert_time_to_pu(1))
print("Viscosity: ", flow.units.viscosity_pu)
print("Timesteps: ", int(flow.units.convert_time_to_lu(20)))
simulation.step(num_steps=int(flow.units.convert_time_to_lu(20)))
np.savetxt('tgv3d_smag_Re1600_Res100_energy.csv', simulation.reporters[0].out, delimiter=';')
np.savetxt('tgv3d_smag_Re1600_Res100_enstrophy.csv', simulation.reporters[1].out, delimiter=';')
with open("nu_smag.txt", "w") as f:
    for s in collision.out:
        f.write(str(s) +"\n")




# collision_net = lt.BGKSmagorinskiCollision_net(lattice,tau=flow.units.relaxation_parameter_lu)
# simulation_net = lt.Simulation(flow,lattice,collision_net,streaming)
# simulation_net.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=1, out=None))
# simulation_net.reporters.append(lt.EnstrophyReporter(lattice=lattice, flow=flow , interval=1, out=None))
# simulation_net.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=100))
# print("Convert 1 timestep in lu to pu: ", flow.units.convert_time_to_pu(1))
# print("Viscosity: ", flow.units.viscosity_pu)
# print("Timesteps: ", int(flow.units.convert_time_to_lu(20)))
# simulation_net.step(num_steps=int(flow.units.convert_time_to_lu(20)))
# np.savetxt('tgv3d_nn_Re1600_Res100_energy.csv', simulation_net.reporters[0].out, delimiter=';')
# np.savetxt('tgv3d_nn_Re1600_Res100_enstrophy.csv', simulation_net.reporters[1].out, delimiter=';')
# with open("nu_nn.txt", "w") as f:
#     for s in collision_net.out:
#         f.write(str(s) +"\n")


print("Finish Testing")




