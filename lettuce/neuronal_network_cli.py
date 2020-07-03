import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from neuronal_network import LBMNet

def trainNet(resolution=100,reynolds=1600,steps_lu=100):
    device='cuda:0'
    net = LBMNet()
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=.01)

    if os.path.isfile('neuronal_network_data.pt'):
        net = torch.load('neuronal_network_data.pt')

    import lettuce
    import lettuce as lt

    lattice = lt.Lattice(lt.D3Q27,device)
    flow = lt.TaylorGreenVortex3D(resolution,reynolds,0.1,lattice)
    collision = lt.SmagorinskyCollision(lattice,tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow,lattice,collision,streaming)
    simulation.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=1, out=None))

    for i in range(steps_lu):
        simulation.step(num_steps=1)

        optimizer.zero_grad()

        input = simulation.collision.S_shear.permute(1,2,3,0)
        target = simulation.collision.tau_eff[...,None]

        out = net(input)
        loss = criterion(out,target)
        loss.backward()

        optimizer.step()

        if i % 50 == 0:
            error = loss.item()
            print(f"[{i + 1:4d}] running. Loss: {error / 1:.10f}")


    torch.save(net, 'neuronal_network_data.pt')
    #np.savetxt('tmp_training.csv', simulation.reporters[0].out, delimiter=';')
    print("Finish Training")
    return

def testNet(resolution=100,reynolds=1600,steps_pu=100):
    import lettuce
    import lettuce as lt
    device='cuda:0'
    lattice = lt.Lattice(lt.D3Q27,device)
    flow = lt.TaylorGreenVortex3D(resolution,reynolds,0.1,lattice)
    streaming = lt.StandardStreaming(lattice)
    collision = lt.SmagorinskyCollision(lattice,tau=flow.units.relaxation_parameter_lu,net='neuronal_network_data.pt')
    #collision = lt.BGKCollision(lattice,tau=flow.units.relaxation_parameter_lu)
    simulation = lt.Simulation(flow,lattice,collision,streaming)
    simulation.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=1, out=None))
    simulation.reporters.append(lt.EnstrophyReporter(lattice=lattice, flow=flow , interval=1, out=None))
    simulation.reporters.append(lt.EnergyReporter(lattice=lattice, flow=flow , interval=100))
    print("Convert 1 timestep in lu to pu: ", flow.units.convert_time_to_pu(1))
    print("Viscosity: ", flow.units.viscosity_pu)
    print("Timesteps: ", int(flow.units.convert_time_to_lu(steps_pu)))
    simulation.step(num_steps=int(flow.units.convert_time_to_lu(steps_pu)))
    np.savetxt('../data/tgv3d_smag_Re1600_Res100_energy.csv', simulation.reporters[0].out, delimiter=';')
    np.savetxt('../data/tgv3d_smag_Re1600_Res100_enstrophy.csv', simulation.reporters[1].out, delimiter=';')
    with open("nu_smag.txt", "w") as f:
        for s in collision.out:
            f.write(str(s) +"\n")
    return

resolution = 100
reynolds = 1600
steps_lu = 1000
steps_pu = 20

#trainNet(resolution=resolution, reynolds=reynolds, steps_lu=steps_lu)
print("Finish training")
testNet(resolution=resolution, reynolds=reynolds, steps_pu=steps_pu)
print("Finish testing")