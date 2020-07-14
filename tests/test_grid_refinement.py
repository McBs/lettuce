import lettuce as lt
import numpy as np
import torch

def f_FineCourse(lattice,tau_f,tau_c,f):
    rho = lattice.rho(f)
    u = lattice.u(f)
    f_eq= lattice.equilibrium(rho, u)
    f_neq = f - f_eq
    f_course=f_eq + 2* (1.0 /flow_f.units.relaxation_parameter_lu)/(1.0 /flow_c.units.relaxation_parameter_lu)* f_neq
    u_neu = lattice.rho(f_course)
    rho_neu = lattice.u(f_course)
    return f_course

def test():
    return []


lattice = lt.Lattice(lt.D3Q27,'cpu',dtype=torch.double)
flow = lt.TaylorGreenVortex3D(120,1600,0.15,lattice)
collision = lt.BGKCollision(lattice,tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
sim = lt.Simulation(flow,lattice,collision,streaming)
#f = sim.f[:, ::2, ::2, ::2]
f = sim.f[:, ::2, ::2, ::2]
u = lattice.u(f)
rho = lattice.rho(f)
p, u_init = flow.initial_solution(flow.grid)

u_init = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u_init))[:, ::2, ::2, ::2]
rho_init = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))[:, ::2, ::2, ::2]

print(u.shape)
print(u_init.shape)
print(u-u_init)
#mlups = sim.step(num_steps=1)
#print(mlups, 'MLUPS')


flow_c = lt.TaylorGreenVortex3D(60,1600,0.15,lattice)
collision_c = lt.BGKCollision(lattice,tau=flow_c.units.relaxation_parameter_lu)
sim_c = lt.Simulation(flow_c,lattice,collision_c,streaming)


rho = lattice.rho(f)
u = lattice.u(f)
f_eq= lattice.equilibrium(rho, u)
f_neq = f - f_eq
#f_c=f_eq[:, ::2, ::2, ::2] + 2* (1.0 /flow.units.relaxation_parameter_lu)/(1.0 /flow_c.units.relaxation_parameter_lu)* f_neq[:, ::2, ::2, ::2]
f_c=f_eq + 2* (1.0 /flow.units.relaxation_parameter_lu)/(1.0 /flow_c.units.relaxation_parameter_lu)* f_neq
#u_neu = lattice.rho(f_course)
#rho_neu = lattice.u(f_course)_
f_c_init = sim_c.f

print(f_c-f_c_init)
u_c = lattice.u(f_c)
rho_c = lattice.rho(f_c)

u_c_init = lattice.u(sim_c.f)
rho_c_init = lattice.rho(sim_c.f)

print(u_c_init-u_c)