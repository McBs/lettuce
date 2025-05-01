import torch
import lettuce as lt
import numpy as np
import h5py
import pickle
import io
import os
from typing import Union, List, Optional
from lettuce import UnitConversion, D2Q9, ExtFlow
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional, Any, Tuple, Union, Callable # For type hinting
import glob # For finding files
import re   # For parsing filenames (optional, can use string splitting too)


__all__ = [
    "Transform", "D2Q9Dellar", "ShiftedSigmoid", "HDF5Reporter", "LettuceDataset", "WVelocity", "TotalPressure",
    "Acoustic", "CharacteristicBoundary", "plotU", "plotRho", "TensorReporter", "TensorDataset", "plot_velocity_density"
]

class Transform:
    """Base class that defines the signature for all moment (and cumulant)
    transforms.
    """

    def __init__(self, stencil: 'Stencil', context: 'Context', names=None):
        self.context = context
        self.names = [f"m{i}" for i in range(stencil.q)]\
            if names is None else names
        self.stencil = stencil

    def __getitem__(self, moment_names):
        if not isinstance(moment_names, tuple):
            moment_names = [moment_names]
        return [self.names.index(name) for name in moment_names]

    def transform(self, f):
        return f

    def inverse_transform(self, m):
        return m

    def equilibrium(self, m: torch.Tensor, flow: 'Flow'):
        """A very inefficient and basic implementation of the equilibrium
        moments.
        """
        warnings.warn(
            "Transform.equilibrium is a poor man's implementation of "
            "the moment equilibrium. Please consider implementing the "
            "equilibrium moments for your transform by hand.",
            InefficientCodeWarning
        )
        f = self.inverse_transform(m)
        feq = flow.equilibrium(flow, flow.rho(None, f), flow.u(None, f))
        return self.transform(feq)

    def einsum(self, equation, fields, *args) -> torch.Tensor:
        """Einstein summation on local fields."""
        inputs, output = equation.split("->")
        inputs = inputs.split(",")
        for i, inp in enumerate(inputs):
            if len(inp) == len(fields[i].shape):
                pass
            elif len(inp) == len(fields[i].shape) - self.stencil.d:
                inputs[i] += "..."
                if not output.endswith("..."):
                    output += "..."
            else:
                assert False, "Bad dimension."
        equation = ",".join(inputs) + "->" + output
        return torch.einsum(equation, fields, *args)

    def mv(self, m, v) -> torch.Tensor:
        """matrix-vector multiplication"""
        return self.einsum("ij,j->i", [m, v])

class D2Q9Dellar(Transform):
    matrix = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 0, -1, 0, 1, -1, -1, 1],
         [0, 0, 1, 0, -1, 1, 1, -1, -1],
         [-3 / 2, 3, -3 / 2, 3, -3 / 2, 3, 3, 3, 3],
         [0, 0, 0, 0, 0, 9, -9, 9, -9],
         [-3 / 2, -3 / 2, 3, -3 / 2, 3, 3, 3, 3, 3],
         [1, -2, -2, -2, -2, 4, 4, 4, 4],
         [0, -2, 0, 2, 0, 4, -4, -4, 4],
         [0, 0, -2, 0, 2, 4, 4, -4, -4]]
    )
    inverse = np.array(
        [[4 / 9, 0, 0, -4 / 27, 0, -4 / 27, 1 / 9, 0, 0],
         [1 / 9, 1 / 3, 0, 2 / 27, 0, -1 / 27, -1 / 18, -1 / 12, 0],
         [1 / 9, 0, 1 / 3, -1 / 27, 0, 2 / 27, -1 / 18, 0, -1 / 12],
         [1 / 9, -1 / 3, 0, 2 / 27, 0, -1 / 27, -1 / 18, 1 / 12, 0],
         [1 / 9, 0, -1 / 3, -1 / 27, 0, 2 / 27, -1 / 18, 0, 1 / 12],
         [1 / 36, 1 / 12, 1 / 12, 1 / 54, 1 / 36, 1 / 54, 1 / 36, 1 / 24,
          1 / 24],
         [1 / 36, -1 / 12, 1 / 12, 1 / 54, -1 / 36, 1 / 54, 1 / 36, -1 / 24,
          1 / 24],
         [1 / 36, -1 / 12, -1 / 12, 1 / 54, 1 / 36, 1 / 54, 1 / 36, -1 / 24,
          -1 / 24],
         [1 / 36, 1 / 12, -1 / 12, 1 / 54, -1 / 36, 1 / 54, 1 / 36, 1 / 24,
          -1 / 24]]
    )
    names = ['rho', 'jx', 'jy', 'Pi_xx', 'Pi_xy', 'PI_yy', 'N', 'Jx', 'Jy']
    supported_stencils = [lt.D2Q9]

    def __init__(self, stencil: 'Stencil', context: 'Context'):
        super(D2Q9Dellar, self).__init__(stencil, context, self.names)
        self.matrix = self.context.convert_to_tensor(self.matrix)
        self.inverse = self.context.convert_to_tensor(self.inverse)

    def transform(self, f):
        return self.mv(self.matrix, f)

    def inverse_transform(self, m):
        return self.mv(self.inverse, m)

    def equilibrium(self, m, flow: 'Flow'):
        warnings.warn("I am not 100% sure if this equilibrium is correct.",
                      ExperimentalWarning)
        meq = torch.zeros_like(m)
        rho = m[0]
        jx = m[1]
        jy = m[2]
        Pi_xx = jx * jx / rho * 9 / 2
        Pi_xy = jx * jy / rho * 9
        Pi_yy = jy * jy / rho * 9 / 2
        meq[0] = rho
        meq[1] = jx
        meq[2] = jy
        meq[3] = Pi_xx
        meq[4] = Pi_xy
        meq[5] = Pi_yy
        return meq
class ShiftedSigmoid(torch.nn.Module):
    def forward(self, x):
        """Apply sigmoid transformation with a shift."""
        return torch.sigmoid(x)

class TotalPressure:

    def __init__(self, context, interval, slices=[slice(None, None), slice(None, None)]):
        self.context = context
        self.interval = interval
        self.out_total = []
        self.t = []
        self.slices = slices

    def __call__(self, simulation: 'Simulation'):
        i = simulation.flow.i
        if i % self.interval == 0:
            out = simulation.flow.incompressible_energy()[self.slices[0],self.slices[1]].mean()
            out = simulation.flow.rho()[0,self.slices[0],self.slices[1]].mean()
            out = simulation.flow.units.convert_density_to_lu(simulation.flow.rho()[0,self.slices[0],self.slices[1]]).mean()
            # out = simulation.flow.incompressible_energy().sum()
            # self.rho_total.append(out)
            # self.rho_total.append((simulation.flow.rho()[0,self.slices[0],self.slices[1]]).sum())
            self.out_total.append(out)
            self.t.append(simulation.flow.units.convert_time_to_pu(i))

class HDF5Reporter:

    def __init__(self, context, flow, interval, t_pu, filebase='./output', metadata=None, slices=None, trainingsdomain=False):
        self.context = context
        self.interval = torch.round(flow.units.convert_time_to_lu(torch.arange(0,t_pu+1e-5, interval)))
        self.interval = [0,1,2]
        self.slices = slices
        self.filebase = filebase
        fs = h5py.File(self.filebase + '.h5', 'w')
        if trainingsdomain:
            fs.attrs["trainingsdomain"] = [trainingsdomain[0].start,trainingsdomain[0].stop,trainingsdomain[1].start,trainingsdomain[1].stop]
        if metadata:
            for attr in metadata:
                fs.attrs[attr] = metadata[attr]
        if slices is not None:
            self.shape = (flow.stencil.q, *[slices[0].stop - slices[0].start, slices[1].stop - slices[1].start])
        else:
            self.shape = (flow.stencil.q, *flow.grid[0].shape)
        fs.create_dataset(name="f",
                          shape=(0, *self.shape),
                          maxshape=(None, *self.shape))
        fs.close()

    @staticmethod
    def _pickle_to_h5(instance):
        bytes_io = io.BytesIO()
        pickle.dump(instance, bytes_io)
        bytes_io.seek(0)
        return np.void(bytes_io.getvalue())

    def __call__(self, simulation: 'Simulation'):
        i = simulation.flow.i
        f = simulation.flow.f
        # if i % self.interval == 0:
        if i in self.interval:
            with h5py.File(self.filebase + '.h5', 'r+') as fs:
                fs["f"].resize(fs["f"].shape[0]+1, axis=0)
                if self.slices is not None:
                    fs["f"][-1, ...] = f[:,*self.slices].cpu()
                else:
                    fs["f"][-1, ...] = f.cpu()
                fs.attrs['data'] = str(fs["f"].shape[0])
                fs.attrs['steps'] = str(i)


class LettuceDataset(torch.utils.data.Dataset):
    """ Custom dataset for HDF5 files in lettuce that can be used by torch's
        dataloader.

    Parameters
    ----------
        filebase : string
            Path to the hdf5 file with annotations.
        transform : class object
            Optional transform to be applied on a f loaded from HDF5 file.
        target : logical operation (True, False)
            Returns also the next dataset[idx + skip_idx_to_target] - default=False
        skip_idx_to_target : integer
            Define which next target dataset is returned if target is True - default=1

    Examples
        --------
        Create a data loader.
        >>> import lettuce as lt
        >>> import torch
        >>> lattice = lt.Lattice(lt.D3Q27, device="cpu")
        >>> dataset_train = lt.LettuceDataset(lattice=lattice,
        >>>              filebase= "./hdf5_output.h5",
        >>>              target=True)
        >>> train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True)
        >>> for (f, target, idx) in train_loader:
        >>>     ...
        """

    def __init__(self, context, filebase, transform=None, target=False, skip_idx_to_target=1):
        super().__init__()
        self.filebase = filebase
        self.transform = transform
        self.target = target
        self.skip_idx_to_target = skip_idx_to_target
        self.fs = h5py.File(self.filebase, "r")
        self.shape = self.fs["f"].shape
        self.keys = list(self.fs.keys())
        self.context = context#self._unpickle_from_h5(self.fs.attrs["flow"]).context

    def __str__(self):
        for attr, value in self.fs.attrs.items():
            if attr in ('flow', '_collision'):
                print(attr + ": " + str(self._unpickle_from_h5(self.fs.attrs[attr])))
            else:
                print(attr + ": " + str(value))
        return ""

    def __len__(self):
        return self.shape[0] - self.skip_idx_to_target if self.target else self.shape[0]

    def __getitem__(self, idx):
        f = self.get_data(idx)
        target = []
        if self.target:
            target = self.get_data(idx + self.skip_idx_to_target)
        if self.transform:
            f = self.transform(f)
            if self.target:
                target = self.transform(target)
        return (f, target, idx) if self.target else (f, idx)

    def __del__(self):
        self.fs.close()

    def __call__(self, idx):
        return self.get_data(idx)

    def get_data(self, idx):
        return self.context.convert_to_tensor(self.fs["f"][idx])

    def get_attr(self, attr):
        return self.fs.attrs[attr]

    @staticmethod
    def _unpickle_from_h5(byte_str):
        return pickle.load(io.BytesIO(byte_str))

class WVelocity(lt.Boundary):
    """Sets distributions on this boundary to equilibrium with predefined
    velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes
    equations. This boundary condition should only be used if no better
    options are available.
    """

    def __init__(self, context: 'Context', mask, velocity, pressure=0):
        velocity = [velocity] if not hasattr(velocity, '__len__') else velocity
        self.velocity = context.convert_to_tensor(velocity)
        self.pressure = context.convert_to_tensor(pressure)
        self._mask = mask

    def __call__(self, flow: 'Flow'):
        f_local = flow.f[:, 0, :]
        rho = (f_local[0]+f_local[2]+f_local[4]+2*(f_local[3]+f_local[7]+f_local[6])) / (1 - flow.units.convert_velocity_to_lu(self.velocity)[0])
        ru = rho * flow.units.convert_velocity_to_lu(self.velocity)[0]

        f_local[1] = f_local[3] + (2/3) * ru
        f_local[5] = f_local[7] + (1/6) * ru + (1/2)*(f_local[4]-f_local[2])
        f_local[8] = f_local[6] + (1/6) * ru + (1/2)*(f_local[4]-f_local[2])
        f_out = flow.f.clone()
        f_out[:, 0, :] = f_local
        return f_out

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

class Acoustic(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_fneq: bool = True,
                 velocity_init = 1,
                 K=None,
                 distanceFromRight=200,
                 xc = 150):
        self.initialize_fneq = initialize_fneq
        self.velocity_init = velocity_init
        self.distanceFromRight = distanceFromRight
        self.K = 0 if K is None else K
        self.xc = xc
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

        xyz = tuple([
            self.units.convert_length_to_pu(torch.arange(startpoints[0],endpoints[0], device=self.context.device, dtype=self.context.dtype)),
            self.units.convert_length_to_pu(torch.arange(startpoints[1],endpoints[1], device=self.context.device, dtype=self.context.dtype))
        ])
            # torch.linspace(startpoints[n], endpoints[n],
                                   # steps=self.resolution[n],
                                   # device=self.context.device,
                                   # dtype=self.context.dtype))

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
        xc = self.xc
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
        mask_outlet = torch.zeros_like(x, dtype=torch.bool);
        mask_outlet[-1, :] = True
        Outlet = CharacteristicBoundary(context=self.context,
                                  mask=mask_outlet,
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
        self.rho_dt_old = context.convert_to_tensor(torch.zeros_like(mask[-1,:]),dtype=float)
        self.u_dt_old = context.convert_to_tensor(torch.zeros_like(mask[-1,:]),dtype=float)
        self.v_dt_old = context.convert_to_tensor(torch.zeros_like(mask[-1,:]),dtype=float)
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
        # K0 = self.K(f_left)[:, 0] if callable(self.K)  else self.K
        K0 = self.K(f_left,self.rho_dt_old,self.u_dt_old,self.v_dt_old)[:, 0] if callable(self.K)  else self.K
        L1 = -K0*(self.cs2*rho_local-self.cs2*1)
        L3 = u_local * v_dx

        rho_dt = -self._inv_two_cs2 * (L5 + L1)
        u_dt = -1/(2 * rho_local * self.cs) * (L5 + L1)
        v_dt = -L3

        # self.rho_t1 = rho_local + 1.5 * rho_dt - 0.5 * self.rho_dt_old
        # self.u_t1 = u_local + 1.5 * u_dt - 0.5 * self.u_dt_old
        # self.v_t1 = v_local + 1.5 * v_dt - 0.5 * self.v_dt_old
        self.rho_t1 = rho_local + rho_dt
        self.u_t1 = u_local + u_dt
        self.v_t1 = v_local + v_dt

        self.rho_dt_old = rho_dt
        self.u_dt_old = u_dt
        self.v_dt_old = v_dt
        # flow.f[:, -1, :] = f_local
        # return flow.f
        f_out = flow.f.clone()
        f_out[:, -1, :] = f_local
        return f_out

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

def plotU(f, flow, slices, config, rectangle = False):
    u = flow.units.convert_velocity_to_pu(flow.u(f)).cpu()
    u_norm = np.linalg.norm(u.detach().numpy(), axis=0)
    plt.imshow(u_norm[slices[0], slices[1]].transpose(), vmin=.985, vmax=1.015, origin='lower')
    if rectangle:
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((config["nx"] - 200, config["ny"] // 2 - 100), 200, 200, fill=None, alpha=1))
    plt.title('Velocity Simulation')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plotRho(f, flow, slices, config, rectangle = False):
    rho = flow.units.convert_density_to_pu(flow.rho(f)).cpu()[0]
    plt.imshow(rho[slices[0], slices[1]].detach().numpy().transpose(), vmin=-1e-4 + 1, vmax=1e-4 + 1, origin='lower')
    if rectangle:
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((config["nx"] - 200, config["ny"] // 2 - 100), 200, 200, fill=None, alpha=1))
    plt.title('Density Simulation')
    plt.colorbar()
    plt.tight_layout()
    plt.show()




class TensorReporter:
    """
    A reporter class that saves the 'f' tensor and optional 'trainingsdomain'
    information from a simulation at specified intervals using torch.save.
    The saving intervals are calculated based on physical time units.
    Data is saved as a dictionary in a .pt file.
    """
    def __init__(self,
                 flow: 'Flow',                       # Added flow object
                 interval: float,                   # Added interval frequency (physical units)
                 t_lu: float,                       # Added total physical time
                 filebase: str = './output',
                 trainingsdomain = None):
        """
        Initializes the TensorReporter.

        Args:
            flow ('Flow'): The simulation flow object, expected to have
                           `flow.units.convert_time_to_lu` method.
            interval (float): The desired saving interval in physical units.
            t_pu (float): The total simulation time in physical units.
            filebase (str): The base path and filename prefix for saving files.
                            The step number and '.pt' extension will be appended.
                            Defaults to './output'.
            trainingsdomain (Optional[Any]): Information about the training domain.
                                             Expected format like the original HDF5 code:
                                             e.g., [slice(start0, stop0), slice(start1, stop1)]
                                             or similar object with .start/.stop attributes.
                                             It will be processed and stored as a flat list:
                                             [start0, stop0, start1, stop1].
                                             If None, it won't be saved. Defaults to None.
        """
        self.filebase = filebase

        # --- Calculate saving interval steps ---
        try:
            # Generate time points in physical units
            # time_points_pu = torch.arange(0, t_pu + 1e-5, interval)
            time_points_lu = torch.arange(0, t_lu, interval)
            # Convert physical time points to simulation steps (lattice units time)
            # interval_steps_tensor = torch.round(flow.units.convert_time_to_lu(time_points_pu))
            interval_steps_tensor = time_points_lu
            # Convert tensor to a list of unique integers
            self.interval = sorted(list(set(map(int, interval_steps_tensor.tolist()))))
            # Ensure step 0 is included if the range starts at 0
            if 0 not in self.interval and t_lu >= 0 and interval > 0:
                 # This logic might need adjustment based on how convert_time_to_lu(0) behaves
                 # If convert_time_to_lu(0) is exactly 0, torch.round might keep it.
                 # If it's slightly off, rounding might miss it. Let's explicitly add if needed.
                 # Check if the first calculated step is > 0 but the time range started at 0
                 if not self.interval or self.interval[0] > 0:
                     self.interval.insert(0, 0)
            # self.interval = [0,1,2]
            print(f"TensorReporter: Calculated save steps: {self.interval}") # Log calculated steps
        except AttributeError as e:
             print(f"Error: Failed to calculate interval steps. Check 'flow' object structure. Expected 'flow.units.convert_time_to_lu'. Details: {e}")
             # Set a default or raise error if calculation is critical
             # self.interval = [0] # Example fallback
             raise # Re-raise by default, as this is likely essential
        except Exception as e:
            print(f"Error during interval calculation: {e}")
            # self.interval = [0] # Example fallback
            raise # Re-raise by default

        # --- Process trainingsdomain ---
        self.processed_trainingsdomain: Optional[List[Union[int, float]]] = None
        if trainingsdomain is not None:
            try:
                # Attempt to convert to the flat list format [start0, stop0, start1, stop1]
                if (len(trainingsdomain) >= 2 and
                        hasattr(trainingsdomain[0], 'start') and hasattr(trainingsdomain[0], 'stop') and
                        hasattr(trainingsdomain[1], 'start') and hasattr(trainingsdomain[1], 'stop')):
                     self.processed_trainingsdomain = [
                         trainingsdomain[0].start, trainingsdomain[0].stop,
                         trainingsdomain[1].start, trainingsdomain[1].stop
                     ]
                elif isinstance(trainingsdomain, (list, tuple)) and len(trainingsdomain) == 4 and all(isinstance(x, (int, float)) for x in trainingsdomain):
                    self.processed_trainingsdomain = list(trainingsdomain)
                else:
                    print(f"Warning: Unexpected 'trainingsdomain' format: {trainingsdomain}. Could not process into [start0, stop0, start1, stop1] list. It will not be saved.")
            except (TypeError, AttributeError, IndexError) as e:
                print(f"Warning: Error processing 'trainingsdomain': {trainingsdomain}. Error: {e}. It will not be saved.")
                self.processed_trainingsdomain = None

    def __call__(self, simulation: 'Simulation'):
        """
        Called during the simulation loop. Saves the tensor 'f' and potentially
        'trainingsdomain' if the current simulation step is in the calculated
        interval list. Data is saved to a .pt file as a dictionary.

        Args:
            simulation ('Simulation'): The simulation object, expected to have
                                       simulation.flow.i (current step) and
                                       simulation.flow.f (tensor to save).
        """
        # Get current step and tensor from simulation object
        try:
            i = simulation.flow.i
            f = simulation.flow.f
        except AttributeError as e:
            print(f"Error: Simulation object structure mismatch. Expected simulation.flow.i and simulation.flow.f. Details: {e}")
            raise # Re-raise the error as it's fundamental

        # Check if the current step 'i' is one of the steps we want to save at
        if i in self.interval:
            # Construct the full filename including the step number and .pt extension
            filename = f"{self.filebase}_{i:06d}.pt"
            print(f"Saving data for step {i} to {filename}")

            # Prepare data to be saved in a dictionary
            data_to_save = {
                'f': f.cpu(), # Save tensor on CPU
                'step': i
            }

            # Add trainingsdomain to the dictionary if it was processed successfully
            if self.processed_trainingsdomain is not None:
                data_to_save['trainingsdomain'] = self.processed_trainingsdomain
                # print(f"  Including trainingsdomain: {self.processed_trainingsdomain}") # Optional log

            # Save the dictionary using torch.save
            try:
                torch.save(data_to_save, filename)
            except Exception as e:
                print(f"Error saving data dictionary to {filename}: {e}")
                # Optionally handle the error (e.g., retry, log details, raise)

class TensorDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for loading simulation data saved by TensorReporter.
    Loads individual .pt files containing dictionaries {'f': tensor, 'step': int, ...}.
    """

    def __init__(self,
                 file_pattern: str,
                 transform: Optional[Callable] = None,
                 target: bool = False,
                 skip_idx_to_target: int = 1,
                 device: Optional[Union[str, torch.device]] = None,
                 sort_key_func: Optional[Callable[[str], int]] = None,
                 verbose: bool = False):
        """
        Initializes the TensorDataset.

        Args:
            file_pattern (str): A pattern (glob compatible) to find the .pt files.
                                Example: './output_data/sim_data_*.pt'
            transform (Optional[Callable]): A function/transform to apply to the loaded 'f' tensors.
                                            Defaults to None.
            target (bool): If True, __getitem__ returns (f, target_f, step).
                           If False, __getitem__ returns (f, step). Defaults to False.
            skip_idx_to_target (int): The number of *indices* (in the sorted file list)
                                      to skip ahead to find the target 'f'. Defaults to 1.
            device (Optional[Union[str, torch.device]]): The device to load tensors onto
                                                         ('cpu', 'cuda', etc.). If None,
                                                         tensors are loaded onto the device
                                                         they were saved from (usually CPU).
                                                         Defaults to None.
            sort_key_func (Optional[Callable[[str], int]]): A function to extract the step
                                                            number (as int) from a filename
                                                            for sorting. If None, it attempts
                                                            to extract the last number before '.pt'.
                                                            Defaults to None.
        """
        super().__init__()
        self.file_pattern = file_pattern
        self.transform = transform
        self.target = target
        self.skip_idx_to_target = skip_idx_to_target
        self.device = device
        self.trainingsdomain = None # Will be loaded from the first file
        self.verbose = verbose

        if sort_key_func is None:
            # Default function to extract step number (assumes format like *_123.pt)
            def default_sort_key(filepath):
                try:
                    # Find numbers just before the '.pt' extension
                    match = re.search(r'_(\d+)\.pt$', os.path.basename(filepath))
                    if match:
                        return int(match.group(1))
                    else:
                        # Fallback or raise error if pattern doesn't match
                        print(f"Warning: Could not extract step number from {filepath}")
                        return -1 # Or raise ValueError
                except Exception as e:
                    print(f"Error extracting step from {filepath}: {e}")
                    return -1 # Or raise
            sort_key_func = default_sort_key

        # 1. Find and sort files
        all_files = glob.glob(self.file_pattern)
        if not all_files:
            raise FileNotFoundError(f"No files found matching pattern: {self.file_pattern}")

        # Create a list of (step_number, filepath) tuples and sort by step_number
        try:
            self.data_files = sorted(
                [(sort_key_func(f), f) for f in all_files],
                key=lambda x: x[0] # Sort by the extracted step number (first element)
            )
            # Filter out files where step extraction failed (returned -1 or similar)
            self.data_files = [item for item in self.data_files if item[0] >= 0]
            if not self.data_files:
                 raise ValueError(f"No valid step numbers could be extracted from files matching {self.file_pattern}")
        except Exception as e:
             raise ValueError(f"Error processing or sorting files matching {self.file_pattern}: {e}")


        if self.verbose: print(f"Found {len(self.data_files)} data files.")

        # 2. Load metadata (like trainingsdomain) from the first file
        try:
            first_step, first_filepath = self.data_files[0]
            if self.verbose: print(f"Loading metadata from first file: {first_filepath} (Step {first_step})")
            # Load to CPU first to avoid potential GPU memory issues during init
            initial_data = torch.load(first_filepath, map_location='cpu')
            if 'trainingsdomain' in initial_data:
                self.trainingsdomain = initial_data['trainingsdomain']
                if self.verbose: print(f"  Loaded trainingsdomain: {self.trainingsdomain}")
            else:
                print("  'trainingsdomain' key not found in the first file.")
            # Can load other metadata here if needed

            # Check tensor shape from first file if useful (optional)
            if 'f' in initial_data:
                 self.tensor_shape = initial_data['f'].shape
                 if self.verbose: print(f"  Tensor 'f' shape from first file: {self.tensor_shape}")
            else:
                 print(f"Warning: 'f' key not found in the first file: {first_filepath}")
                 self.tensor_shape = None

        except IndexError:
            print("Warning: No data files available to load metadata from.")
        except KeyError as e:
             print(f"Warning: Key {e} not found while loading metadata from {first_filepath}.")
        except Exception as e:
            print(f"Error loading metadata from {first_filepath}: {e}")
            # Decide if this is critical - maybe raise the error depending on requirements
            # raise e


    def __len__(self) -> int:
        """Returns the number of available starting points."""
        num_files = len(self.data_files)
        if self.target:
            # We need enough files for both the input and the target
            return max(0, num_files - self.skip_idx_to_target)
        else:
            return num_files

    def __call__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Allows calling the dataset instance like a function to get an item by index."""
        return self.__getitem__(idx)
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int]]:
        """
        Loads and returns data for a given index.

        Args:
            idx (int): The index in the sorted list of found files.

        Returns:
            If self.target is False: (f_tensor, step_number)
            If self.target is True: (f_tensor, target_f_tensor, step_number)

        Raises:
            IndexError: If idx or the calculated target index is out of bounds.
            FileNotFoundError: If a specific .pt file is missing unexpectedly.
            KeyError: If a loaded dictionary is missing the 'f' key.
        """
        if not 0 <= idx < len(self):
             # This check considers the reduced length when self.target is True
             raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")

        # --- Get data for the current index ---
        current_step, current_filepath = self.data_files[idx]
        try:
            current_data = torch.load(current_filepath, map_location=self.device)
            f = current_data['f']
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {current_filepath} (for index {idx})")
        except KeyError:
            raise KeyError(f"Key 'f' not found in {current_filepath}")
        except Exception as e:
             raise RuntimeError(f"Error loading {current_filepath}: {e}")

        # --- Apply transform to current data ---
        if self.transform:
            f = self.transform(f)

        # --- Handle target if requested ---
        if self.target:
            target_idx = idx + self.skip_idx_to_target
            # We already checked bounds in __len__ and the initial idx check,
            # so target_idx should be valid if idx is valid for the target case.
            # However, an extra assertion can be helpful for debugging.
            assert 0 <= target_idx < len(self.data_files), f"Target index {target_idx} calculation error."

            target_step, target_filepath = self.data_files[target_idx]
            try:
                target_data = torch.load(target_filepath, map_location=self.device)
                target_f = target_data['f']
            except FileNotFoundError:
                raise FileNotFoundError(f"Target data file not found: {target_filepath} (for target index {target_idx})")
            except KeyError:
                raise KeyError(f"Key 'f' not found in target file {target_filepath}")
            except Exception as e:
                 raise RuntimeError(f"Error loading target file {target_filepath}: {e}")

            # --- Apply transform to target data ---
            if self.transform:
                target_f = self.transform(target_f)

            return f, target_f, current_step # Return current step number
        else:
            return f, current_step # Return current step number

    def get_trainingsdomain(self) -> Optional[List[Union[int, float]]]:
        """Returns the trainingsdomain loaded from the first data file."""
        return self.trainingsdomain

    # __del__ is generally not needed as file handles aren't kept open.
    # def __del__(self):
    #     pass

    # __call__ could mimic __getitem__ if desired, but often not needed for Datasets
    # def __call__(self, idx):
    #     return self.__getitem__(idx)

    # get_attr is less relevant as attributes are per-file. Use get_trainingsdomain() instead.
    # def get_attr(self, attr):
    #    if attr == 'trainingsdomain':
    #        return self.get_trainingsdomain()
    #    # Handle other potential attributes if needed, perhaps loading from first file?
    #    raise AttributeError(f"Attribute '{attr}' not directly available.")


def plot_velocity_density(f, flow, slices, config, rectangle=False, figsize=(12, 5)):
    """
    Plots velocity magnitude and density side-by-side.

    Args:
        f: Input tensor containing distribution functions.
        flow: Flow object with methods u(), rho(), and units conversion.
        slices: Tuple of slices to apply to the spatial dimensions for plotting.
        config: Dictionary containing configuration like "nx", "ny".
        rectangle (bool): If True, draws a rectangle on both plots.
        figsize (tuple): Figure size for the combined plot.
    """
    # Create a figure and a set of subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Plot 1: Velocity ---
    ax_u = axes[0] # Left subplot
    u = flow.units.convert_velocity_to_pu(flow.u(f)).cpu()
    u_norm = np.linalg.norm(u.detach().numpy(), axis=0)

    # Display the velocity magnitude image on the left axes
    im_u = ax_u.imshow(u_norm[slices[0], slices[1]].transpose(),
                       vmin=0.985,  # Use vmin/vmax from original plotU
                       vmax=1.015,
                       origin='lower',
                       aspect='auto') # 'auto' or 'equal' depending on desired aspect ratio
    ax_u.set_title('Velocity Magnitude')
    fig.colorbar(im_u, ax=ax_u) # Add colorbar associated with the left axes

    if rectangle:
        # Add rectangle to the left axes
        rect_u = Rectangle((config["nx"] - 200, config["ny"] // 2 - 100),
                           200, 200, fill=None, alpha=1, edgecolor='red') # Added edgecolor for visibility
        ax_u.add_patch(rect_u)

    # --- Plot 2: Density ---
    ax_rho = axes[1] # Right subplot
    rho = flow.units.convert_density_to_pu(flow.rho(f)).cpu()[0]

    # Display the density image on the right axes
    im_rho = ax_rho.imshow(rho[slices[0], slices[1]].detach().numpy().transpose(),
                           vmin=-1e-4 + 1, # Use vmin/vmax from original plotRho
                           vmax=1e-4 + 1,
                           origin='lower',
                           aspect='auto') # Match aspect ratio if desired
    ax_rho.set_title('Density')
    fig.colorbar(im_rho, ax=ax_rho) # Add colorbar associated with the right axes

    if rectangle:
        # Add rectangle to the right axes
        rect_rho = Rectangle((config["nx"] - 200, config["ny"] // 2 - 100),
                             200, 200, fill=None, alpha=1, edgecolor='red') # Added edgecolor for visibility
        ax_rho.add_patch(rect_rho)

    # --- Final Adjustments ---
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()       # Show the entire figure with both plots

# Example Usage (replace with your actual objects):
# Assuming you have f, flow, slices, config defined
# plot_velocity_density(f_tensor, your_flow_object, your_slices, your_config, rectangle=True)