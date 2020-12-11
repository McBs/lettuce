"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""


import torch
import numpy as np
from lettuce.util import torch_gradient


__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum"]


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
        return vorticity * dx**self.lattice.D


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""
    def __init__(self, lattice, flow):
        super(EnergySpectrum, self).__init__(lattice, flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)
        self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2 if self.lattice.D == 3 else self.dimensions[0] / self.dx
        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
            (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
            (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[...,0]**2 + uh[...,1]**2), dim=0)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek


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
        mass = f[...,1:-1,1:-1].sum()
        if self.mask is not None:
            mass -= (f*self.mask.to(dtype=torch.float)).sum()
        return mass

class TurbulenceStatistics(Observable):
    """Turbulence statistics in physical units.

    Parameters
    ----------
    taylor_microscale : torch.Tensor
    reynolds_number_lambda : torch.Tensor
    autocorrelation_fct
    length_scales
    """
    def __init__(self, lattice, flow, taylor_microscale=True,
                 reynolds_number_lambda=True, autocorrelation_fct=False, length_scales=True):
        super(TurbulenceStatistics, self).__init__(lattice, flow)
        self.taylor_microscale = taylor_microscale
        self.reynolds_number_lambda = reynolds_number_lambda
        self.autocorrelation_fct = autocorrelation_fct
        self.length_scales = length_scales
        self.Dim = self.lattice.D

    def __call__(self, f):

        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        if self.lattice.D == 2:
            uf = u - torch.mean(u, dim=tuple(np.arange(1, self.lattice.D + 1)))[..., None, None]
        if self.lattice.D == 3:
            uf = u - torch.mean(u, dim=tuple(np.arange(1,self.lattice.D+1)))[..., None, None, None]
        r = int(self.flow.units.characteristic_length_lu / 2)
        R = [torch.mean(uf * torch.roll(uf, i, dims=1), dim=tuple(np.arange(1,self.lattice.D+1))) /
             torch.mean(uf ** 2, dim=tuple(np.arange(1,self.lattice.D+1))) for i in range(r)]
        R = torch.cat(R).reshape(r,self.lattice.D)

        f_ddr = (-R[2,0] + 16 * R[1,0] - 30 * R[0,0] + 16 * R[1,0] - R[2,0]) / (12 * self.flow.units.convert_length_to_pu(1) ** 2)
        lambda_f = 1 / torch.sqrt(-.5 * f_ddr)

        if self.lattice.D == 2:
            uf_RMS = torch.sqrt(1 / 2 * (torch.mean(uf[0] ** 2 + uf[1] ** 2)))
        if self.lattice.D == 3:
            uf_RMS = torch.sqrt(1 / 3 * (torch.mean(uf[0] ** 2 + uf[1] ** 2 + uf[2] ** 2)))
        Re_l = uf_RMS * lambda_f / self.flow.units.viscosity_pu

        statistics = torch.stack([self.flow.units.convert_length_to_lu(lambda_f), lambda_f, Re_l, uf_RMS, torch.sum(R[:,0]),torch.sum(R[:,1])])

        return statistics