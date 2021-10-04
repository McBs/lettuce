"""
DecayingTurbulence vortex in 2D and 3D. Dimension is set by the stencil.
Special Inputs & standard value: wavenumber_energy-peak = 20, initial_energy = 0.5

Additional attributes / properties
__________
energy_spectrum: returns a pair [spectrum, wavenumbers]
"""

import numpy as np
from lettuce.unit import UnitConversion
from lettuce.grid import RegularGrid
from lettuce import mpiClass

class DecayingTurbulence:

    def __init__(self, resolution, reynolds_number, mach_number, lattice, k0=20, ic_energy=0.5,mpiObject=None):
        self.k0 = k0
        self.ic_energy = ic_energy
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=None
        )
        self.ref=0
       
        self.mpiObject=lattice.mpiObject
        self.wavenumbers = []
        self.spectrum = []
        if(lattice.D==2):
            self.rgrid = RegularGrid([resolution, resolution], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False,mpiObject=self.mpiObject,lattice=lattice)
        else:
            self.rgrid = RegularGrid([resolution, resolution, resolution], self.units.characteristic_length_lu,
                                self.units.characteristic_length_pu, endpoint=False,mpiObject=self.mpiObject,lattice=lattice)

    def refinment(self,newResolution):
        
        self.resolution=newResolution
        reynolds_number=self.units.reynolds_number
        mach_number=self.units.mach_number
        resolution=newResolution
        lattice=self.units.lattice
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution, characteristic_length_pu=2 * np.pi,
            characteristic_velocity_pu=None
        )
        p, u = self.initial_solution(0)
        self.ref=1

    def analytic_solution(self, x, t=0):
        return

    def _generate_wavenumbers(self):
        self.dimensions = self.grid[0].shape
        frequencies = [np.fft.fftfreq(dim, d=1 / dim) for dim in self.dimensions]
        wavenumber = np.meshgrid(*frequencies)
        wavenorms = np.linalg.norm(wavenumber, axis=0)
        self.wavenumbers = np.arange(int(np.max(wavenorms)))
        wavemask = (wavenorms[..., None] > self.wavenumbers - 0.5) & (wavenorms[..., None] <= self.wavenumbers + 0.5)
        return wavenorms, wavenumber, wavemask

    def _generate_spectrum(self):
        wavenorms, wavenumber, wavemask = self._generate_wavenumbers()
        ek = (wavenorms) ** 4 * np.exp(-2 * (wavenorms / self.k0) ** 2)
        ek /= np.sum(ek)
        ek *= self.ic_energy
        self.spectrum = ek[..., None] * wavemask
        self.spectrum = np.sum(self.spectrum, axis=tuple((np.arange(self.units.lattice.D))))
        return ek, wavenumber

    def _generate_initial_velocity(self, ek, wavenumber):
        dx = self.units.convert_length_to_pu(1.0)
        inputrand=np.array(wavenumber).shape
        randomres=np.random.random(inputrand)
        erg=randomres * 2 * np.pi + 0j
        u = erg
        u = [np.fft.fftn(u[dim], axes=tuple((np.arange(self.units.lattice.D)))) for dim in range(self.units.lattice.D)]

        u_real = [u[dim].real for dim in range(self.units.lattice.D)]
        u_imag = [u[dim].imag for dim in range(self.units.lattice.D)]
        for dim in range(self.units.lattice.D):
            u_real[dim].ravel()[0] = 0
            u_imag[dim].ravel()[0] = 0

        u_real_h = [np.sqrt(2 / self.units.lattice.D * ek / (u_imag[dim] ** 2 + u_real[dim] ** 2 + 1.e-15))
                    * u_real[dim] for dim in range(self.units.lattice.D)]
        u_imag_h = [np.sqrt(2 / self.units.lattice.D * ek / (u_imag[dim] ** 2 + u_real[dim] ** 2 + 1.e-15))
                    * u_imag[dim] for dim in range(self.units.lattice.D)]
        for dim in range(self.units.lattice.D):
            u_real_h[dim].ravel()[0] = 0
            u_imag_h[dim].ravel()[0] = 0

        ### Remove divergence
        # modified wave number sin(k*dx) is used, as the gradient below uses second order cental differences
        # Modify if other schemes are used or use kx, ky if you don't know the modified wavenumber !!!
        wavenumber_modified = [np.sin(wavenumber[dim] * dx) / dx for dim in range(self.units.lattice.D)]
        wavenorm_modified = np.linalg.norm(wavenumber_modified, axis=0) + 1e-16

        divergence_real = np.zeros(self.dimensions)
        divergence_imag = np.zeros(self.dimensions)
        for dim in range(self.units.lattice.D):
            divergence_real += wavenumber_modified[dim] * u_real_h[dim]
            divergence_imag += wavenumber_modified[dim] * u_imag_h[dim]

        u_real = [u_real_h[dim] - divergence_real * wavenumber_modified[dim]
                  / wavenorm_modified ** 2 for dim in range(self.units.lattice.D)]
        u_imag = [u_imag_h[dim] - divergence_imag * wavenumber_modified[dim]
                  / wavenorm_modified ** 2 for dim in range(self.units.lattice.D)]
        for dim in range(self.units.lattice.D):
            u_real[dim].ravel()[0] = 0
            u_imag[dim].ravel()[0] = 0

        ### Scale velocity field to achieve the desired inicial energy
        e_kin = [np.sum(u_real[dim] ** 2 + u_imag[dim] ** 2) for dim in range(self.units.lattice.D)]
        e_kin = np.sum(e_kin) * .5

        factor = np.sqrt(self.ic_energy / e_kin)
        u_real = [u_real[dim] * factor for dim in range(self.units.lattice.D)]
        u_imag = [u_imag[dim] * factor for dim in range(self.units.lattice.D)]

        ### Backtransformation to physical space
        norm = ((self.resolution * dx ** (1 - self.units.lattice.D) * np.sqrt(self.units.characteristic_length_pu))
                if self.units.lattice.D == 3 else (self.resolution / dx))

        u = np.asarray([
            (np.fft.ifftn(u_real[dim] + u_imag[dim] * 1.0j, axes=tuple((np.arange(self.units.lattice.D)))) * norm).real
            for dim in range(self.units.lattice.D)])

        return u

    def _compute_initial_pressure(self):
        return np.zeros(self.dimensions)[None, ...]

    def initial_solution(self, x):
        """Return initial solution. Note: this function sets the characteristic velocity in phyiscal units."""
        ek, wavenumber = self._generate_spectrum()
        u = self._generate_initial_velocity(ek, wavenumber)
        p = self._compute_initial_pressure()
        self.units.characteristic_velocity_pu = np.linalg.norm(u, axis=0).max()
        return p, u

    @property
    def energy_spectrum(self):
        return self.spectrum, self.wavenumbers

    @property
    def grid(self):
        #grid = [np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False) for _ in range(self.units.lattice.D)] #anpassen
        x=np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        y=np.linspace(0, 2 * np.pi, num=self.resolution, endpoint=False)
        if(self.ref==1):
            index=self.mpiObject.index
            x =x[index.start:index.stop,...]
        y = np.linspace(0., 1., num=self.resolution, endpoint=False)
        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        return []
