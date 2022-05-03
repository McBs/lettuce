import torch
import torch.distributed as dist
import torchdd as dd
import lettuce as lt
from lettuce.flows.flow import Flow
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dist.init_process_group(backend="mpi", rank=-1, world_size=-1)


res = 50
time = 1 #sec
step = 2
device = "cpu"
interval = 50
re = 400

grid_points_per_D_in_X = 14
grid_points_per_D_in_Y = 10
grid_points_per_D_in_Z = 3
D = 10

position_1 = [(int(round(5*D + D * 1.06, 0)),0,0)]
position_2 = [(int(round(5*D + D * 1.54, 0)),0,0)]
position_3 = [(int(round(5*D + D * 2.02, 0)),0,0)]
time_start_recording = 1


class cylinder3D(Flow):
    """Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """

    def __init__(self, domain, reynolds_number, mach_number, lattice, char_length_lu, compute_f):
        self.domain = domain
        self.units = lt.UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=1,
            characteristic_velocity_pu=1)
        super().__init__(domain=domain,
                         units=self.units,
                         compute_f=compute_f)
        self._mask = np.zeros(shape=self.domain.shape, dtype=np.bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.domain.shape
        self._mask = m.astype(np.bool)

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0, 0.0])[..., None, None, None]
        u = (1 - self.mask.astype(np.float)) * u_char
        return p, u

    @property
    def boundaries(self):
        if self.domain.rank == 0:
            boundaries = [
                lt.EquilibriumBoundaryPU(np.abs(self.domain.grid()[0]) == 0, self.units.lattice, self.units,
                                      np.array([self.units.characteristic_velocity_pu, 0, 0])),
                lt.BounceBackBoundary(self.mask, self.units.lattice)]
        elif self.domain.rank ==  dist.get_world_size():
            boundaries = [
                lt.AntiBounceBackOutlet(self.units.lattice, [1, 0, 0]),
                lt.BounceBackBoundary(self.mask, self.units.lattice)]
        else:
            boundaries = [
                lt.BounceBackBoundary(self.mask, self.units.lattice)]
        return boundaries

class AveragedVelocityReporter:
    """Reports the streamwise velocity averaged in span direction (z) at x=x_row"""
    def __init__(self, lattice, flow, position, interval=1, starting_step=1, out=None):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.starting_step = starting_step
        self.out = [] if out is None else out
        self.is_on_rank =(self.flow.domain.contains(torch.tensor(position)))
        print(self.is_on_rank)
        if self.is_on_rank is True:
            self.x_row = self.flow.domain.contains(torch.tensor(position))
            print(self.flow.domain.grid()[0][:,0,0])
            print(self.x_row)
    def __call__(self, i, t, f):
        if self.is_on_rank is True:
            u = self.lattice.u(f)[:,self.x_row,:,:]
            u = self.flow.units.convert_velocity_to_pu(u).cpu().numpy()
            entry = np.mean(u,axis=2)
        else:
            entry = None
        return entry


dtype = torch.float64
domain = dd.BoxDomain(
    lower=torch.zeros(3),
    upper=10*torch.tensor([grid_points_per_D_in_X,
                        grid_points_per_D_in_Y,
                        grid_points_per_D_in_Z]),
    resolution=torch.Size([grid_points_per_D_in_X*D,
                           grid_points_per_D_in_Y*D,
                           grid_points_per_D_in_Z*D]),
    n_ghost=[[0, 0], [0, 0], [0, 0]],
    mpi_rank=0,
    device="cpu",
    dtype=torch.float64,
    endpoint=True)
if domain.rank == 0:
    print(domain)
print(f"number of cells: {domain.n_cells}")

lattice_cpu = lt.Lattice(lt.D3Q27, device="cpu", dtype=dtype)
lattice_gpu = lt.Lattice(lt.D3Q27, device=device+":"+str(dist.get_rank()), dtype=dtype)

flow = cylinder3D(lattice=lattice_gpu,
                  domain=domain,
                  mach_number=0.05,
                  reynolds_number=re,
                  char_length_lu=D,
                  compute_f=False)

decom = dd.DomainDecomposition(domain=domain,
                               flow=flow,
                               dims=[1,1,1],
                               mpi=True)
domains = decom.split_domain()
for i in domains:
    print(i)


flows = cylinder3D(lattice=lattice_gpu,
                   domain=domains[0],
                   mach_number=0.05,
                   reynolds_number=re,
                   char_length_lu=D,
                   compute_f=False)
flows.units = flow.units

for i in domains:
    print("I'm ",flows.__class__.__name__," on rank:", i.rank," with ",flows.domain,", coordinates:", flows.domain.coord)


mask_np = np.zeros(domains[0].shape,dtype=bool)

#The center of the cylinder is located at the following coordinates
center_x = 5*D
center_y = int(grid_points_per_D_in_Y/2*D)

# Set the mask for the cylinder (Note that bounce back boundaries set the wall between the fluid and the boundary node)
for X in np.arange(domains[0].shape[0]):
    for Y in np.arange(domains[0].shape[1]):
        if ((X-center_x)**2 + (Y-center_y)**2) <= (((D-1)/2)**2):
            mask_np[X,Y,:] = True

boundary_points_mask = np.zeros_like(mask_np)
boundary_angle_info = np.ones_like(mask_np,dtype=float)*1000
flows.mask = mask_np



collision = lt.BGKCollision(lattice_gpu, tau=flows.units.relaxation_parameter_lu)
streaming = dd.MPIStreaming(lattice=lattice_gpu, decom=decom, device=device)
simulation = lt.Simulation(flow=flows, lattice=lattice_gpu,  collision=collision, streaming=streaming)
Velocity0 = AveragedVelocityReporter(lattice_gpu, flow, position_1, 1, int(flow.units.convert_time_to_lu(time_start_recording)))
simulation.reporters.append(Velocity0)
#
# energy = lt.IncompressibleKineticEnergy(lattice_gpu, flow)
# reporter = dd.MPIObservableReporter(energy, decomposition=decom, interval=interval,)
# simulation.reporters.append(reporter)
#
steps = int(flow.units.convert_time_to_lu(time)) if step is None else step
if dist.get_rank() == 0:
    print("steps:",steps)
print(f"Start simulation on rank: {domains[0].rank}")
mlups = simulation.step(steps)
print(f"Finish with {mlups} MLUPS")
# simulation.save_checkpoint(filename="test"+str(domains[0].rank))

