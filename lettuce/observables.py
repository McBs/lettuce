"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""

import torch
import numpy as np
from lettuce.util import torch_gradient
from packaging import version

__all__ = [
    "Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum",
    "Correlation", "U_max_lu", "U_rms", "Dissipation_sij", "Turbulent_kinetic_energy", "TimeCorrelation",
    "Dissipation_E_pu", "Skewness", "Flatness", "PDF"
           ]


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError


class MaximumVelocity(Observable):
    """Maximum velocitiy"""

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.flow.units.convert_velocity_to_pu(torch.norm(u, dim=0).max())


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE


class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """

    def __call__(self, f):
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx ** self.lattice.D


class EnergySpectrum(Observable):
    """
    Calculates the energy spectrum of a velocity using the Fast Fourier Transform (FFT).

    Args:
        f (torch.Tensor): Population.

    Returns:
        ek (torch.Tensor): Energy spectrum according to Pope (Eq. 6.193).

    Notes:
        - This function is applicable for a three-dimensional flow only.
        - This function is applicable for an isotropic turbulence simulation only.
        - The function is normed for a physical characteristic length of 2π.
          For further normalization options see: https://github.com/fdietzsc/hita/tree/master
        - Conditions may be defined within initialization process, which has been omitted due to memory constraints.
    """

    def __init__(self, lattice, flow):
        super(EnergySpectrum, self).__init__(lattice, flow)
        assert lattice.D == 3, "This is not a three-dimensional flow."
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        self.wavenorms = torch.norm(torch.stack(torch.meshgrid(*frequencies)), dim=0)
        wavenumbers = torch.arange(self.dimensions[0])
        self.wavemask = (
                (self.wavenorms[..., None] > wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
                (self.wavenorms[..., None] <= wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )
        self.k = (torch.arange(0, int(self.dimensions[0] / 2)) + 1)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        return self.spectrum_from_u(self.flow.units.convert_velocity_to_pu(self.lattice.u(f)))

    def spectrum_from_u(self, u: torch.Tensor) -> torch.Tensor:
        # Computes the N dimensional discrete Fourier transform of the velocity
        uh = torch.stack([
            torch.abs(torch.fft.fftn(u[i], dim=tuple(torch.arange(self.lattice.D)), norm="backward"))*self.dimensions[0]**-3 for i in
            range(self.lattice.D)
        ])

        # Compute the values of spectrum elementwise
        ekin = torch.sum(((uh) ** 2), dim=0)
        spectrum = torch.zeros_like(self.k, dtype=self.lattice.dtype)
        counter = torch.zeros_like(self.k, dtype=self.lattice.dtype)

        # Calculate values of specturm (for wavenumbers [k0+1,kmax-1])
        for nr, k in enumerate(self.k[1:-1]):
            condition = (self.wavenorms <= (float(k) + 0.5)) & (self.wavenorms > (float(k) - 0.5))
            spectrum[nr + 1] = ekin[condition].sum()
            counter[nr + 1] = condition.sum()

        # Calculate first value of spectrum (for the wavenumber k0)
        condition = (self.wavenorms <= (self.k[0] + 0.5))
        spectrum[0] = ekin[condition].sum()
        counter[0] = condition.sum()

        # Calculate last value of spectrum (for the wavenumber kmax)
        condition = (self.wavenorms <= self.k[-1]) & (self.wavenorms > (self.k[-1] - 0.5))
        spectrum[-1] = ekin[condition].sum()
        counter[-1] = condition.sum()

        # Norm the spectrum with respect to the spherical shell
        return spectrum * (2 * np.pi) * (self.k) ** 2 / (counter)


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundaries).
    """

    def __init__(self, lattice, flow, no_mass_mask=None):
        super(Mass, self).__init__(lattice, flow)
        self.mask = no_mass_mask

    def __call__(self, f):
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()
        return mass

class U_max_lu(Observable):

    def __init__(self, lattice, flow):
        super(U_max_lu, self).__init__(lattice, flow)

    def __call__(self, f):
        return (torch.abs(self.lattice.u(f)).max())

class U_rms(Observable):

    def __init__(self, lattice, flow):
        super(U_rms, self).__init__(lattice, flow)

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.flow.units.convert_velocity_to_pu(((u**2).mean())**0.5)

class Skewness(Observable):

    def __init__(self, lattice, flow):
        super(Skewness, self).__init__(lattice, flow)

    def __call__(self, f):
        U = self.lattice.u(f)
        u = U - torch.mean(U, dim=(1, 2, 3))[:, None, None, None]
        # du = torch.stack([torch_gradient(u[_], order=6)[_] for _ in range(3)])
        sk = torch.mean(u ** 3, dim=(1, 2, 3)) / (torch.mean(u ** 2, dim=(1, 2, 3)) ** (3 / 2))
        return sk

class Flatness(Observable):

    def __init__(self, lattice, flow):
        super(Flatness, self).__init__(lattice, flow)

    def __call__(self, f):
        U = self.lattice.u(f)
        u = U - torch.mean(U, dim=(1, 2, 3))[:, None, None, None]
        # du = torch.stack([torch_gradient(u[_])[_] for _ in range(3)])
        return torch.mean(u**4, dim=(1, 2, 3)) / torch.mean(u**2, dim=(1, 2, 3))**(2)

class Turbulent_kinetic_energy(Observable):

    def __init__(self, lattice, flow):
        super(Turbulent_kinetic_energy, self).__init__(lattice, flow)

    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        return 0.5 * torch.mean(self.lattice.einsum("i,i->",[u,u]))

class Correlation(Observable):
    """AutoCorrelation function to calculate longitudinal and transversal correlations

    Notes
    -----
    The output existing of the longitudinal R11/f(r) and transversal R22/g(r) correlation
    is concatenated in the first dimension.
    Example:
    f(r).shape = torch.Size([16])
    g(r).shape = torch.Size([16])
    output.shape = torch.Size([32])

    Parameters
    ----------

    """

    def __init__(self, lattice, flow, fft=True):
        super(Correlation, self).__init__(lattice, flow)
        self.fft = fft
    def __call__(self, f):
        return self.calculate_correlation_fft(self.lattice.u(f)) if self.fft else self.calculate_correlation(self.lattice.u(f))

    def calculate_correlation_fft(self, u: torch.float):
        u_fft = (torch.stack([
            (torch.fft.fftn(u[i], dim=tuple(torch.arange(u.shape[0])))) for i in range(u.shape[0])
        ]))

        Rij = (u_fft * torch.conj(u_fft))
        R = (torch.stack([
            (torch.fft.ifftn(Rij[i], dim=tuple(torch.arange(u.shape[0])))) for i in range(u.shape[0])
        ]))
        RR = R / (torch.std(u, dim=[1, 2, 3])[..., None, None, None] ** 2) / u.shape[1] ** 3
        R11 = (RR[0, :, 0, 0] + RR[1, 0, :, 0] + RR[2, 0, 0, :]) / 3

        R1_22 = (RR[0, 0, :, 0] + RR[2, 0, :, 0]) / 2
        R2_22 = (RR[1, :, 0, 0] + RR[2, :, 0, 0]) / 2
        R3_22 = (RR[0, 0, 0, :] + RR[1, 0, 0, :]) / 2
        R22 = (R1_22 + R2_22 + R3_22) / 3
        return torch.cat([R11[:int(self.flow.resolution/2)].real, R22[:int(self.flow.resolution/2)].real])

    def calculate_correlation(self, u: torch.float):
        r = torch.arange(u.shape[1] / 2).byte()
        R11_0 = torch.zeros_like(r).to(dtype=torch.float64)
        R11_1 = torch.zeros_like(r).to(dtype=torch.float64)
        R11_2 = torch.zeros_like(r).to(dtype=torch.float64)
        R22_0 = torch.zeros_like(r).to(dtype=torch.float64)
        R22_1 = torch.zeros_like(r).to(dtype=torch.float64)
        R22_2 = torch.zeros_like(r).to(dtype=torch.float64)

        vel = u - u.mean()
        for index, i in enumerate(r):
            R11_0[index] = (vel[0] * torch.roll(vel[0], shifts=[index, 0, 0], dims=[0, 1, 2])).mean() / (vel[0] ** 2).mean()
            R11_1[index] = (vel[1] * torch.roll(vel[1], shifts=[0, index, 0], dims=[0, 1, 2])).mean() / (vel[1] ** 2).mean()
            R11_2[index] = (vel[2] * torch.roll(vel[2], shifts=[0, 0, index], dims=[0, 1, 2])).mean() / (vel[2] ** 2).mean()

            R22_0[index] = (
                               (vel[1] * torch.roll(vel[1], shifts=[index, 0, 0], dims=[0, 1, 2])).mean() / (
                                   vel[1] ** 2).mean() +
                               (vel[2] * torch.roll(vel[2], shifts=[index, 0, 0], dims=[0, 1, 2])).mean() / (
                                           vel[2] ** 2).mean()
                           ) / 2
            R22_1[index] = (
                               (vel[0] * torch.roll(vel[0], shifts=[0, index, 0], dims=[0, 1, 2])).mean() / (
                                   vel[0] ** 2).mean() +
                               (vel[2] * torch.roll(vel[2], shifts=[0, index, 0], dims=[0, 1, 2])).mean() / (
                                           vel[2] ** 2).mean()
                           ) / 2
            R22_2[index] = (
                               (vel[0] * torch.roll(vel[0], shifts=[0, 0, index], dims=[0, 1, 2])).mean() / (
                                   vel[0] ** 2).mean() +
                               (vel[1] * torch.roll(vel[1], shifts=[0, 0, index], dims=[0, 1, 2])).mean() / (
                                           vel[1] ** 2).mean()
                           ) / 2

        R11 = (R11_0 + R11_1 + R11_2) / 3
        R22 = (R22_0 + R22_1 + R22_2) / 3
        return torch.cat([R11, R22])


class TimeCorrelation(Observable):
    """TimeCorrelation function to calculate longitudinal and transversal correlations

    Notes
    -----

    """

    def __init__(self, lattice: "Lattice", flow: "Flow"):
        super(TimeCorrelation, self).__init__(lattice, flow)
        self.u_init = None

    def __call__(self, f):
        correlation = self.correlation(self.flow.units.convert_velocity_to_pu(self.lattice.u(f)))
        return correlation

    def correlation(self, u):
        vel = u - u.mean([1, 2, 3])[..., None, None, None]
        if self.u_init is None:
            self.u_init = vel
            self.u_norm = (self.u_init ** 2).mean()
        return (self.u_init * vel).mean() * self.u_norm ** -1

class Dissipation_sij(Observable):

    def __init__(self, lattice, flow, no_grad=True):
        super(Dissipation_sij, self).__init__(lattice, flow)
        self.no_grad = no_grad
    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        dx = self.flow.units.convert_length_to_pu(1.0)
        nu = self.flow.units.viscosity_pu

        u_ij = torch.stack([torch_gradient(u[i], dx=dx, order=6,no_grad=self.no_grad) for i in range(self.lattice.D)])
        s_ij = 0.5 * (u_ij + torch.transpose(u_ij, 0, 1))
        dissipation = 2 * nu * torch.mean((s_ij ** 2).sum(0).sum(0))
        return dissipation

class Dissipation_E_pu(Observable):
    def __init__(self, lattice, flow, nu=None, no_grad=True):
        super(Dissipation_E_pu, self).__init__(lattice, flow)
        self.no_grad = no_grad
        self.nu = nu
        self.k = torch.arange(0, int(flow.resolution/2)) + 1
    def __call__(self, Ek):
        dx = 1
        return torch.sum(2*self.nu*self.k**2*Ek)*dx

class PDF(Observable):
    def __init__(self, lattice, flow, resolution=250, range=[-7,7]):
        super(PDF, self).__init__(lattice, flow)
        self.resolution = resolution
        self.range = range
        self.bins = np.linspace(range[0],range[1],resolution+1,endpoint=True)
        self.bins = self.bins[:-1]+0.5*(self.bins[1]-self.bins[0])
    def __call__(self, f):
        u = self.lattice.u(f)
        u = (u - torch.mean(u, dim=(1, 2, 3))[:, None, None, None]).ravel().cpu()
        a = torch.histogram(u / torch.std(u), bins=self.resolution, range=(self.range[0], self.range[1]), density=False)[0]
        return  a / len(u)
