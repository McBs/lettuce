"""
datautils for writing/reading hdf5 files.
"""

import h5py

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