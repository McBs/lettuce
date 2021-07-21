"""
datautils for writing/reading hdf5 files.
"""

import h5py
from torch.utils import data
from lettuce.util import *

class hdf5writer:
    def __init__(self, flow, lattice, collision, interval, filebase='./output'):
        self.lattice = lattice
        self.interval = interval
        self.filebase = filebase
        fs = h5py.File(self.filebase + '.hdf5', 'w')
        fs.attrs['device'] = self.lattice.device
        fs.attrs['dtype'] = str(self.lattice.dtype)
        fs.attrs['stencil'] = self.lattice.stencil.__name__
        fs.attrs["flow"] = flow.__class__.__name__
        fs.attrs["relaxation_parameter"] = flow.units.relaxation_parameter_lu
        fs.attrs['resolution'] = flow.resolution
        fs.attrs['reynolds_number'] = flow.units.reynolds_number
        fs.attrs['mach_number'] = flow.units.mach_number
        fs.attrs['collision'] = collision.__class__.__name__
        fs.close()
        self.shape =tuple(
            j for i in (flow.units.lattice.Q, flow.grid[0].shape) for j in (i if isinstance(i, tuple) else (i,)))

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            with h5py.File(self.filebase + '.hdf5', 'r+') as fs:
                dset = fs.create_dataset(f"{i:06d}", self.shape)
                dset[:] = self.lattice.convert_to_numpy(f)


class LettuceDataset(data.Dataset):
    """ Custom dataset for HDF5 files in lettuce.

    Parameters
    ----------
        filebase : string
            Path to the hdf5 file with annotations.
        transform : class object
            Optional transform to be applied on a f loaded from HDF5 file.
        target : logical operation (True, False)
            Returns also the next dataset[idx+interval] - default=False
        interval : integer
            Define which next target dataset is returned - default=1
    """

    def __init__(self, lattice, filebase, transform=None, target=False, interval=1):
        super().__init__()
        self.filebase = filebase
        self.lattice = lattice
        self.transform = transform
        self.target = target
        self.interval = 1
        self.keys = []
        self._get_key_info()

    def __len__(self):
        with h5py.File(self.filebase, "r") as fs:
            return len(fs) - self.interval if self.target else len(fs)

    def __getitem__(self, idx):
        f = self.get_data(idx)
        target = []
        if self.target:
            target = self.get_data(idx + self.interval)
        if self.transform:
            f = self.transform(f)
            if self.target:
                target = self.transform(target)
        return (f, target, idx) if self.target else (f, idx)

    def get_data(self, idx):
        with h5py.File(self.filebase, "r") as fs:
            f = self.lattice.convert_to_tensor(fs[self.keys[idx]][:])
        return f

    def get_attr(self, attr):
        with h5py.File(self.filebase, "r") as fs:
            attr = fs.attrs[attr]
        return attr

    def get_attrs_info(self):
        with h5py.File(self.filebase, "r") as fs:
            for attr, value in fs.attrs.items():
                print(attr + ": " + str(value))

    def _get_key_info(self):
        with h5py.File(self.filebase, "r") as fs:
            self.keys = list(fs.keys())




