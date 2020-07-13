import lettuce as lt
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lettuce.neuronal_network import LBMNet
import random

def shuffle_data():
    return

def f_FineCourse(lattice,flow_f,flow_c,f):
    rho = lattice.rho(f)
    u = lattice.u(f)
    f_eq= lattice.equilibrium(rho, u)
    f_neq = f - f_eq
    f_course=f_eq + 2* (1.0 /flow_f.units.relaxation_parameter_lu)/(1.0 /flow_c.units.relaxation_parameter_lu)* f_neq
    u_neu = lattice.rho(f_course)
    rho_neu = lattice.u(f_course)
    return f_course


"""
Setup Net
-----
"""
device='cuda:0'
net = LBMNet()
net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=.01)


"""
Setup simulation
-----
"""
lattice = lt.Lattice(lt.D3Q27, 'cuda')
streaming = lt.StandardStreaming(lattice)
flow_f = lt.TaylorGreenVortex3D(resolution=120, reynolds_number=1600, mach_number=0.05, lattice=lattice)
flow_c = lt.TaylorGreenVortex3D(resolution=60, reynolds_number=1600, mach_number=0.05, lattice=lattice)

collision = lt.SmagorinskyCollision(lattice, tau=flow_f.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow=flow_f, lattice=lattice, collision=collision, streaming=streaming)

collision_c = lt.SmagorinskyCollision_nn(lattice, tau=flow_c.units.relaxation_parameter_lu, net=net)
simulation_c = lt.Simulation(flow=flow_c, lattice=lattice, collision=collision_c, streaming=streaming)


"""
Simulation parameters
-----
"""
#print(flow.units.viscosity_pu)
#print(flow.units.convert_time_to_pu(1))
#print(int(flow.units.convert_time_to_lu(20)))

time = 20
time_lu = flow_f.units.convert_time_to_lu(time)
data = int(flow_f.units.convert_time_to_lu(time)/1000)
print("Viscosity: {}.".format(flow_f.units.viscosity_pu))
print("Tau: {}.".format(flow_f.units.relaxation_parameter_lu))
print("One step in physical units: {}.".format(flow_f.units.convert_time_to_pu(1)))
print("Required timesteps for {}sec: {}.".format(time, int(flow_f.units.convert_time_to_lu(time))))
print("Data volume: {}".format(data))

data = 100
f_course = np.empty([data, 27, 60, 60, 60])
u_course = np.empty([data, 3, 60, 60, 60])
q_course = np.empty([data, 1, 60, 60, 60])
data_input_fine = torch.empty(data,31,60,60,60,device="cpu",requires_grad=True)
data_output_fine = torch.empty(data,31,60,60,60,device="cpu",requires_grad=True)
data_output_course = torch.empty(data,31,60,60,60,device="cpu",requires_grad=True)
data2_output_course = torch.empty(data,1,60,60,60,device="cpu",requires_grad=True)
data2_output_fine = torch.empty(data,1,60,60,60,device="cpu",requires_grad=True)

for i in range(data):
    simulation.step(num_steps=98)
    # f_course[i,...] = lattice.convert_to_numpy(f_FineCourse(lattice, flow_f, flow_c, simulation.f[:,::2,::2,::2]))
    with torch.no_grad():
        data_input_fine[i,0:27,...] = f_FineCourse(lattice, flow_f, flow_c, simulation.f[:,::2,::2,::2])
        data_input_fine[i, 27:30, ...] = lattice.u(simulation.f[:,::2,::2,::2])
        data_input_fine[i, 30, ...] =lattice.rho(simulation.f[:,::2,::2,::2])
    simulation.step(num_steps=2)
    # u_course[i,...] = lattice.convert_to_numpy(lattice.u(simulation.f[:,::2,::2,::2]))
    # q_course[i,...] = lattice.convert_to_numpy(lattice.rho(simulation.f[:,::2,::2,::2]))
    with torch.no_grad():
        data_output_fine[i, 0:27, ...] = f_FineCourse(lattice, flow_f, flow_c, simulation.f[:, ::2, ::2, ::2])
        data_output_fine[i, 27:30, ...] = lattice.u(simulation.f[:, ::2, ::2, ::2])
        data_output_fine[i, 30, ...] = lattice.rho(simulation.f[:, ::2, ::2, ::2])
        data2_output_fine[i,...] = simulation.collision.tau_eff[None,::2, ::2, ::2]
    print(i)


z = 0
for i in random.sample(range(0,data),data):
    z+=1
    print(z)
    simulation_c.f = data_input_fine[i,0:27,...].cuda()
    simulation_c.step(num_steps=1)
    with torch.no_grad():
        data_output_course[i, 0:27, ...] = simulation_c.f
        data_output_course[i, 27:30, ...] = lattice.u(simulation_c.f)
        data_output_course[i, 30, ...] = lattice.rho(simulation_c.f)
        data2_output_course[i, ...] = simulation_c.collision.tau_eff[None,...]

    optimizer.zero_grad()
    # input = lattice.u(simulation_c.f).permute(1,2,3,0)
    # input.requires_grad=True
    # target = lattice.convert_to_tensor(u_course[i,...]).permute(1,2,3,0)
    # target.requires_grad=True

    # loss = criterion(input,target)
    loss = criterion(data_output_course[i,...].permute(1,2,3,0),data_output_fine[i,...].permute(1,2,3,0))
    # loss = criterion(data2_output_course[i, ...].permute(1, 2, 3, 0), data2_output_fine[i, ...].permute(1, 2, 3, 0))
    loss.backward()
    optimizer.step()

    error = loss.item()
    print(f"[{i + 1:4d}] running. Loss: {error / 1:.10f}")


torch.save(net, 'neuronal_network_data.pt')
print("Done with d'shit")

