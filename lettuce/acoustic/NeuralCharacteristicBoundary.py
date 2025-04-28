import torch
import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Union, List, Optional
from lettuce import UnitConversion, D2Q9, ExtFlow
from matplotlib.patches import Rectangle
from utility import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from plot import *
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import torch.optim as optim

class Acoustic(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_fneq: bool = True,
                 velocity_init = 1,
                 K=None,
                 distanceFromRight=200):
        self.initialize_fneq = initialize_fneq
        self.velocity_init = velocity_init
        self.distanceFromRight = distanceFromRight
        self.K = 0 if K is None else K
        if stencil is None and not isinstance(resolution, list):
            warnings.warn("Requiring information about dimensionality!"
                          " Either via stencil or resolution. Setting "
                          "dimension to 2.", UserWarning)
            self.stencil = D2Q9()
        else:
            self.stencil = stencil() if callable(stencil) else stencil
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * self.stencil.d
        else:
            assert len(resolution) in [2, 3], ('the resolution of a '
                                               'taylor-green-vortex '
                                               'must be 2- or 3-dimensional!')
            return resolution

    def make_units(self, reynolds_number, mach_number,
                   resolution) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=200,
            characteristic_length_pu=10,
            characteristic_velocity_pu=1)

    @property
    def grid(self):
        endpoints = self.resolution
        endpoints = [self.distanceFromRight,  self.resolution[1]]
        startpoints = [self.distanceFromRight - self.resolution[0], 0]

        xyz = tuple(self.units.convert_length_to_pu(torch.linspace(startpoints[n], endpoints[n],
                                   steps=self.resolution[n],
                                   device=self.context.device,
                                   dtype=self.context.dtype))
                    for n in range(self.stencil.d))
        return torch.meshgrid(*xyz, indexing='ij')

    def initial_pu(self) -> (torch.Tensor, torch.Tensor):
        p = torch.zeros((1, *self.resolution),
                        device=self.context.device,
                        dtype=self.context.dtype)
        u = torch.full(self.resolution,
                       fill_value=self.velocity_init,
                       device=self.context.device,
                       dtype=self.context.dtype)
        v = torch.zeros(self.resolution,
                        device=self.context.device,
                        dtype=self.context.dtype)
        U = torch.stack([u, v], dim=0)
        p, U = self.convectedVortex()
        return p, U

    def convectedVortex(self) -> (torch.Tensor, torch.Tensor):
        xc, yc = [r * 0.5 for r in self.resolution]
        xc = 100
        # yc = 100
        x, y = self.grid  # beide Shape: (nx, ny)
        x = self.units.convert_length_to_lu(x)
        y = self.units.convert_length_to_lu(y)
        ux0 = self.units.convert_velocity_to_lu(self.velocity_init)
        beta = 0.5
        Rc = 20.0
        gamma = 0.5
        Cv = 1.0 / 3.0

        r2 = (x - xc) ** 2 + (y - yc) ** 2
        r = torch.sqrt(r2)

        d = torch.pow(
            1.0 - (beta * ux0) ** 2 / (2.0 * Cv)
            * torch.exp(1.0 - r2),
            1.0 / (gamma - 1.0)
        )

        exp_term = torch.exp(-r2 / (2.0 * Rc))
        u_x = (ux0
            - beta * ux0 * (y - yc) / Rc * exp_term)
        u_y = beta * ux0 * (x - xc) / Rc * exp_term

        p = self.units.convert_density_lu_to_pressure_pu(d)
        u_x = self.units.convert_velocity_to_pu(u_x)
        u_y = self.units.convert_velocity_to_pu(u_y)
        U = torch.stack([u_x, u_y], dim=0)
        return p, U

    @property
    def boundaries(self) -> List['Boundary']:
        x = self.grid[0]
        Inlet = WVelocity(context=self.context,
                                  mask=torch.abs(x) < 1e-6,
                                  velocity=[1, 0]
                                  )
        Outlet = CharacteristicBoundary(context=self.context,
                                  mask=torch.abs(x) >= 10,
                                  velocity=[self.units.convert_velocity_to_lu(self.velocity_init), 0],
                                  K=self.K,
                                  mach=self.units.mach_number)
        return [Inlet, Outlet]

class CharacteristicBoundary(lt.Boundary):
    """Sets distributions on this boundary to equilibrium with predefined
    velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes
    equations. This boundary condition should only be used if no better
    options are available.
    """

    def __init__(self, context: 'Context', mask, velocity, pressure=0, K=0, mach=None):
        velocity = [velocity] if not hasattr(velocity, '__len__') else velocity
        self.velocity = context.convert_to_tensor(velocity)
        self.pressure = context.convert_to_tensor(pressure)
        self._mask = mask
        self.rho_dt_old = context.convert_to_tensor(0)
        self.u_dt_old = context.convert_to_tensor(0)
        self.v_dt_old = context.convert_to_tensor(0)
        self.rho_t1 = context.convert_to_tensor(1)
        self.u_t1 = context.convert_to_tensor(self.velocity[0])
        self.v_t1 = context.convert_to_tensor(0)
        self.K = K
        self.mach = mach
        self.cs = context.convert_to_tensor(np.sqrt(1 / 3))
        self.cs2 = context.convert_to_tensor(1 / 3)
        self._inv_two_cs2 = context.convert_to_tensor(1 / (2 * self.cs2))
        self._three_half = context.convert_to_tensor(1.5)

    def __call__(self, flow: 'Flow'):
        f_local = flow.f[:,-1,:]
        f_left = flow.f[:,-2,:]
        rho_t1 = self.rho_t1
        u_t1 = self.u_t1
        v_t1 = self.v_t1
        feq = flow.equilibrium(flow, rho_t1, torch.stack([u_t1,v_t1])) #To be adjusted

        f_local[0] = f_local[0] + rho_t1 - 1/(1+u_t1) * (f_local[0] + f_local[2] + f_local[4]
                                                  + 2 * (f_local[1] + f_local[5] + f_local[8]))
        f_local[6] = feq[6] + (f_local[8]-feq[8])+0.5*((f_local[4]-feq[4])-(f_local[2]-feq[2]))
        f_local[3] = feq[3] + (f_local[1]-feq[1])
        f_local[7] = feq[7] + (f_local[5]-feq[5])-0.5*((f_local[4]-feq[4])-(f_local[2]-feq[2]))

        rho_left = f_left.sum(0)
        rho_local = f_local.sum(0)
        u_left = (f_left[1]-f_left[3]+f_left[5]-f_left[6]-f_left[7]+f_left[8])/rho_left
        u_local = (f_local[1]-f_local[3]+f_local[5]-f_local[6]-f_local[7]+f_local[8])/rho_local
        v_left = (f_left[2]-f_left[4]+f_left[5]+f_left[6]-f_left[7]-f_left[8])/rho_left
        v_local = (f_local[2]-f_local[4]+f_local[5]+f_local[6]-f_local[7]-f_local[8])/rho_local

        p_dx = -(1/3 * (rho_left - rho_local))
        u_dx = -(u_left-u_local)
        v_dx = -(v_left-v_local)

        L5 = (u_local + self.cs) * (p_dx + rho_local * self.cs * u_dx)
        K0 = self.K(f_left, self.mach)[:, 0] if callable(self.K)  else self.K
        L1 = -K0*(self.cs2*rho_local-self.cs2*1)
        L3 = u_local * v_dx

        rho_dt = -self._inv_two_cs2 * (L5 + L1)
        u_dt = -1/(2 * rho_local * self.cs) * (L5 + L1)
        v_dt = -L3

        self.rho_t1 = rho_local + 1.5 * rho_dt - 0.5 * self.rho_dt_old
        self.u_t1 = u_local + 1.5 * u_dt - 0.5 * self.u_dt_old
        self.v_t1 = v_local + 1.5 * v_dt - 0.5 * self.v_dt_old

        self.rho_dt_old = rho_dt
        self.u_dt_old = u_dt
        self.v_dt_old = v_dt
        flow.f[:, -1, :] = f_local
        # f_out = flow.f.clone()
        # f_out[:, -1, :] = f_local
        return flow.f

    def make_no_collision_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        pass

    def make_no_streaming_mask(self, shape: List[int], context: 'Context'
                               ) -> Optional[torch.Tensor]:
        pass

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int) -> 'NativeBoundary':
        return None

def run(context, config, K, dataset, dataset_nr, t_pu):
    # with torch.no_grad():
    flow = Acoustic(context, [config["nx"]+config["extension"], config["ny"]],
                    reynolds_number=config["Re"],
                    mach_number=config["Ma"],
                    velocity_init=1,
                    K=K,
                    distanceFromRight=200+config["extension"])
    collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
    if config["save_dataset"]:
        print(f"Saving dataset for Mach {config["Ma"]:03.2f} every {config["save_iteration"]:2.2f} seconds")
        hdf5_reporter = HDF5Reporter(
                     flow=flow,
                     context=context,
                     interval= int(flow.units.convert_time_to_lu(config["save_iteration"])),
                     filebase=f"./dataset_mach-{config["Ma"]:03.2f}_interv-{config["save_iteration"]:03.2f}")
        simulation.reporter.append(hdf5_reporter)
    if config["load_dataset"] and dataset_nr is not None and callable(dataset_train):
        simulation.flow.f = dataset(dataset_nr)[:,:config["nx"]+config["extension"],:config["ny"]]
    with torch.set_grad_enabled(config["train"]):
        simulation(num_steps=int(flow.units.convert_time_to_lu(t_pu)))
    return flow

class NeuralTuning(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device='cuda', nodes=40, index=None):
        """Initialize a neural network boundary model."""
        super(NeuralTuning, self).__init__()
        self.moment = D2Q9Dellar(lt.D2Q9(), lt.Context(device="cuda", dtype=torch.float64, use_native=False))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, nodes, bias=True),
            torch.nn.Linear(nodes, 1, bias=True),
            torch.nn.Sigmoid(),
        ).to(dtype=dtype, device=device)
        self.index = index
        print("Initialized NeuralTuning")

    def forward(self, f, mach):
        """Forward pass through the network with residual connection."""
        mach_tensor = torch.full(
            (f.size(1), 1),
            mach,
            dtype=f.dtype,
            device=f.device
        )
        return (self.net( torch.cat([mach_tensor,f.transpose(0,1)], dim=1)))


if __name__ == "__main__":

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=900)
    parser.add_argument("--extension", type=int, default=0)
    parser.add_argument("--Re", type=int, default=750, help="")
    parser.add_argument("--Ma", type=float, default=0.3, help="")
    parser.add_argument("--t_pu", type=float, default=5)
    parser.add_argument("--load_dataset", action="store_true", default=False)
    parser.add_argument("--load_dataset_idx", type=int, default=0)
    parser.add_argument("--save_dataset", action="store_true", default=False)
    parser.add_argument("--save_iteration", type=float, default=0.25)
    parser.add_argument("--K", type=str, default="neural")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--load_model", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--train_mach_numbers", type = float, nargs = "+", default = [0.15])
    parser.add_argument("--train_t_pu_intervals", type=int,  nargs="+", default=[4])
    parser.add_argument("--expand_intervals", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    args = vars(args)
    [print(arg, args[arg]) for arg in args]
    shift = 7
    torch.manual_seed(0)

    K_tuned = NeuralTuning() if args["K"] == "neural" else 1
    if args["load_model"]:
        K_tunes = torch.load("model_training_v1.pt", weights_only=False)
        print("YES")
    context = lt.Context(torch.device("cuda:0"), use_native=False, dtype=torch.float64)
    slices = [slice(args["nx"] - 200, args["nx"]-1), slice(args["ny"] // 2 - 100, args["ny"] // 2 + 100)]
    # slices = [slice(None, None), slice(None, None)]

    machNumbers = args["train_mach_numbers"]
    intervals = args["train_t_pu_intervals"]
    if args["expand_intervals"]:
        intervals = [x for x in intervals for _ in range(x)]

    pairs = list(product(intervals, machNumbers)) if args["train"] else [(1,args["Ma"])]
    print("Configurations: ", len(pairs))
    if callable(K_tuned) and args["train"]:
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(K_tuned.parameters(), lr=args["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args["scheduler_step"], gamma=0.1)
        epoch_training_loss = []
        scaler = GradScaler()
        optimizer.zero_grad()

    for _ in range(args["epochs"] if args["train"] else 1):
        print(f"Epoch: {_}" if args["train"] else "Running ...")
        running_loss = 0.0
        for i, (idx, ma) in enumerate(pairs):

            dataset_name = f"./dataset_mach-{ma:03.2f}_interv-{args["save_iteration"]:03.2f}.h5"
            dataset_train = (LettuceDataset(context=context, filebase=dataset_name, target=False)
                             if args["load_dataset"] or args["train"] else None)
            if args["train"]: optimizer.zero_grad()

            t_pu = idx * args["save_iteration"] if args["train"] else args["t_pu"]
            print(i, ma, t_pu, idx)
            with autocast(context.device.type):
                flow = run(context=context,
                           config=args,
                           K=K_tuned,
                           dataset = dataset_train,
                           dataset_nr = args["load_dataset_idx"],
                           t_pu = t_pu
                           )
            if callable(K_tuned) and args["train"]:
                reference = dataset_train(idx+args["load_dataset_idx"])[:,slices[0],slices[1]]
                rho_ref = flow.rho(reference)
                rho_train = flow.rho()[:,slices[0],slices[1]]
                # loss = criterion(flow.f[:,slices[0],slices[1]], reference)
                loss = criterion(rho_ref, rho_train)
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print("running_loss:", running_loss)
                # if i == len(pairs)-1:
                #     rho = flow.rho_pu.cpu()[0]
                #     plt.imshow(rho[slices[0],slices[1]].detach().numpy().transpose(), vmin=-4e-5 + 1, vmax=4e-5 + 1,
                #                origin='lower')
                #     currentAxis = plt.gca()
                #     currentAxis.add_patch(
                #         Rectangle((args["nx"] - 200, args["ny"] // 2 - 100), 200, 200, fill=None, alpha=1))
                #     plt.title('Density after simulation')
                #     plt.colorbar()
                #     plt.tight_layout()
                #     plt.show()

        scheduler.step()
        if args["train"]: epoch_training_loss.append(running_loss)
        if args["train"]: print(epoch_training_loss)


    u = flow.units.convert_velocity_to_pu(flow.u()).cpu()
    u_norm = np.linalg.norm(u.detach().numpy(), axis=0)
    half_ny = args["ny"] // 2
    y_start, y_end = half_ny - 100, half_ny + 100
    slices = [slice(None, None), slice(None, None)]
    x_slice = slices[0]
    y_slice = slices[1]

    rectangle_x_slice = slice(-200, None)
    rectangle_y_slice = slice(y_start, y_end)
    # plt.imshow(u_norm[x_slice,y_slice].transpose(), vmin=.985, vmax=1.015, origin='lower')
    # currentAxis = plt.gca()
    # currentAxis.add_patch(Rectangle((args["nx"]-200,args["ny"]//2-100), 200, 200, fill=None, alpha=1))
    # plt.title('Velocity after simulation')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    # rho = flow.rho_pu.cpu()[0]
    # plt.imshow(rho[x_slice,y_slice].detach().numpy().transpose(), vmin=-1.5e-5+1, vmax=1.5e-5+1, origin='lower')
    # currentAxis = plt.gca()
    # currentAxis.add_patch(Rectangle((args["nx"]-200,args["ny"]//2-100), 200, 200, fill=None, alpha=1))
    # plt.title('Velocity after simulation')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    #
    # if args["train"]:
    #     plt.imshow(flow.units.convert_density_to_pu(flow.rho(reference)).detach().cpu().numpy().transpose(), vmin=-1.5e-5+1, vmax=1.5e-5+1, origin="lower")
    #     plt.show()
    #
    #     u_norm = np.linalg.norm(flow.units.convert_velocity_to_pu(flow.u(reference)).detach().cpu().numpy(), axis=0)
    #     plt.imshow(u_norm.transpose(), vmin=.985, vmax=1.015, origin="lower")
    #     plt.title('Velocity after simulation')
    #     plt.colorbar()
    #     plt.tight_layout()
    #     plt.show()
    if args["train"]:
        plot = PlotNeuralNetwork(base="./", show=True, style="./ecostyle.mplstyle")
        plot.loss_function(np.array(epoch_training_loss)/epoch_training_loss[0])
