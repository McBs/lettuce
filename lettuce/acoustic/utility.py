import torch
import lettuce as lt
import numpy as np
import h5py
import pickle
import io
from typing import Union, List, Optional

__all__ = [
    "Transform", "D2Q9Dellar", "ShiftedSigmoid", "HDF5Reporter", "LettuceDataset", "WVelocity", "TotalPressure"
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

    def __init__(self, context, flow, interval, filebase='./output', metadata=None, slices=None):
        self.context = context
        self.interval = interval
        self.slices = slices
        self.filebase = filebase
        fs = h5py.File(self.filebase + '.h5', 'w')
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

    def __call__(self, simulation: 'Simulation'):
        i = simulation.flow.i
        f = simulation.flow.f
        if i % self.interval == 0:
            with h5py.File(self.filebase + '.h5', 'r+') as fs:
                fs["f"].resize(fs["f"].shape[0]+1, axis=0)
                if self.slices is not None:
                    fs["f"][-1, ...] = self.context.convert_to_ndarray(f[:,*self.slices])
                else:
                    fs["f"][-1, ...] = self.context.convert_to_ndarray(f)
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