"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""

import torch
import numpy as np
from lettuce.util import torch_gradient
from packaging import version

__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum",
           "IncompressibleKineticEnergyBd","Dissipation_sij","Dissipation_TGV","SymmetryReporter","EnergySpectrum2",
           "SymmetryTopPercentageReporter","WallQuantities", "GlobalMeanUXReporter"]


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


class IncompressibleKineticEnergyBd(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f):

        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)[1:-1,1:-1]))
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
    """The kinetic energy spectrum"""

    def __init__(self, lattice, flow):
        super(EnergySpectrum, self).__init__(lattice, flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)

        if self.lattice.D == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        #self.wavenumbers = torch.arange(int(torch.max(wavenorms)))

        self.wavenumbers = torch.arange(0, int(self.dimensions[0] / 2))

        #self.wavemask = (
        #        (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
        #        (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        #)

        self.wavemask = (
                (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
                (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        ekin = self._ekin_spectrum(u)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek

    def _ekin_spectrum(self, u):
        """distinguish between different torch versions"""
        torch_ge_18 = (version.parse(torch.__version__) >= version.parse("1.8.0"))
        if torch_ge_18:
            return self._ekin_spectrum_torch_ge_18(u)
        else:
            return self._ekin_spectrum_torch_lt_18(u)

    def _ekin_spectrum_torch_lt_18(self, u):
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[..., 0] ** 2 + uh[..., 1] ** 2), dim=0)
        return ekin

    def _ekin_spectrum_torch_ge_18(self, u):
        uh = (torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.lattice.D))) for i in range(self.lattice.D)
        ]) / self.norm)
        ekin = torch.sum(0.5 * (uh.imag ** 2 + uh.real ** 2), dim=0)
        return ekin


class EnergySpectrum2(Observable):
    """
    Calculates the energy spectrum of a velocity using the Fast Fourier Transform (FFT).

    Args:
        f (torch.Tensor): Population.

    Returns:
        ek (torch.Tensor): Energy spectrum according to Pope (Eq. 6.193).
    """

    def __init__(self, lattice, flow):
        super(EnergySpectrum2, self).__init__(lattice, flow)
        assert lattice.D == 3, "This is not a three-dimensional flow."
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        self.wavenorms = torch.norm(torch.stack(torch.meshgrid(*frequencies)), dim=0)
        wavenumbers = torch.arange(self.dimensions[0])

        # Normierungsfaktor wie in der EnergySpectrum-Klasse
        if self.lattice.D == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        self.wavemask = (
            (self.wavenorms[..., None] > wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
            (self.wavenorms[..., None] <= wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )
        self.k = (torch.arange(0, int(self.dimensions[0] / 2)) + 1).to(dtype=lattice.dtype, device=lattice.device)

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        return self.spectrum_from_u(self.flow.units.convert_velocity_to_pu(self.lattice.u(f)))

    def spectrum_from_u(self, u: torch.Tensor) -> torch.Tensor:
        # Computes the N-dimensional discrete Fourier transform of the velocity
        uh = torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.lattice.D)), norm="backward") / self.norm
            for i in range(self.lattice.D)
        ])

        # Compute the values of spectrum elementwise
        ekin = torch.sum((uh.real ** 2 + uh.imag ** 2), dim=0)
        spectrum = torch.zeros_like(self.k, dtype=self.lattice.dtype)
        counter = torch.zeros_like(self.k, dtype=self.lattice.dtype)

        # Calculate values of spectrum (for wavenumbers [k0+1, kmax-1])
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
        return spectrum * (2 * np.pi) * (self.k ** 2) / counter



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

class Dissipation_sij(Observable):

    def __init__(self, lattice, flow, no_grad=True):
        super(Dissipation_sij, self).__init__(lattice, flow)

    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        dx = self.flow.units.convert_length_to_pu(1.0)
        nu = self.flow.units.viscosity_pu

        u_ij = torch.stack([torch_gradient(u[i], dx=dx, order=6) for i in range(self.lattice.D)])
        s_ij = 0.5 * (u_ij + torch.transpose(u_ij, 0, 1))
        dissipation = 2 * nu * torch.mean((s_ij ** 2).sum(0).sum(0))

        #du_dx=torch.gradient(u[0], dx=dx, order=6)
        #du_dy=torch.gradient(u[1], dx=dx, order=6)
        #du_dz=torch.gradient(u[2], dx=dx, order=6)
        #dissipation= 2*nu*(du_dx**2+du_dy**2+du_dz**2)
        return dissipation

class Dissipation_TGV(Observable):

    def __init__(self, lattice, flow, no_grad=True):
        super(Dissipation_TGV, self).__init__(lattice, flow)

    def __call__(self, f):
        

        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        nges=u.size()[1]



        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)).clone()
        dx = self.flow.units.convert_length_to_pu(1.0)
        nu = self.flow.units.viscosity_pu

        u_new = torch.zeros(3, nges+6, nges+6, nges+6)

        u_new[:, 3:-3, 3:-3, 3:-3] = u

        u_new[:, 0:3, 3:-3, 3:-3] = torch.flip(u[:, 0:3, :, :], [1])
        u_new[0, 0:3, 3:-3, 3:-3] = -1 * u_new[0, 0:3, 3:-3, 3:-3]

        u_new[0, -3:, 3:-3, 3:-3] = -1 * torch.flip(torch.transpose(u[1, :, -3:, :], 0, 1), [0])
        u_new[1, -3:, 3:-3, 3:-3] = torch.flip(torch.transpose(u[0, :, -3:, :], 0, 1), [0])
        u_new[2, -3:, 3:-3, 3:-3] = torch.flip(torch.transpose(u[2, :, -3:, :], 0, 1), [0])

        u_new[:, 3:-3, 0:3, 3:-3] = torch.flip(u[:, :, 0:3, :], [2])
        u_new[1, 3:-3, 0:3, 3:-3] = -1 * u_new[1, 3:-3, 0:3, 3:-3]

        u_new[0, 3:-3, -3:, 3:-3] = torch.flip(torch.transpose(u[1, -3:, :, :], 0, 1), [1])
        u_new[1, 3:-3, -3:, 3:-3] = -1 * torch.flip(torch.transpose(u[0, -3:, :, :], 0, 1), [1])
        u_new[2, 3:-3, -3:, 3:-3] = torch.flip(torch.transpose(u[2, -3:, :, :], 0, 1), [1])

        u_new[:, 3:-3, 3:-3, -3:] = torch.flip(u[:, :, :, -3:], [3])
        u_new[2, 3:-3, 3:-3, -3:] = -1 * u_new[2, 3:-3, 3:-3, -3:]

        u_new[0, 3:-3, 3:-3, 0:3] = torch.flip(torch.transpose(u[1, :, :, 0:3], 0, 1), [2])
        u_new[1, 3:-3, 3:-3, 0:3] = torch.flip(torch.transpose(u[0, :, :, 0:3], 0, 1), [2])
        u_new[2, 3:-3, 3:-3, 0:3] = -1 * torch.flip(torch.transpose(u[2, :, :, 0:3], 0, 1), [2])

        u_grad = torch.stack([torch_gradient(u_new[i], dx=dx, order=6) for i in range(self.lattice.D)])
        u_ij=u_grad[:,:,3:-3,3:-3,3:-3]
        #dissipation=nu*1/nges**3*torch.sum(torch.square(u_ij))
        grad_u0=u_ij[:,0,:,:,:]
        grad_u1=u_ij[:,1,:,:,:]
        grad_u2=u_ij[:,2,:,:,:]
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0])+(grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + (grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
        #u_ij = torch.stack([torch_gradient(u[i], dx=dx, order=6) for i in range(self.lattice.D)])
        s_ij = 0.5 * (u_ij + torch.transpose(u_ij, 0, 1))
        dissipation = 2 * nu * torch.mean((s_ij ** 2).sum(0).sum(0))


        enstrophy= nu*vorticity * dx ** self.lattice.D
        return torch.stack([dissipation, enstrophy])

class SymmetryReporter(Observable):

    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        u = self.lattice.u(f)

        n = u.size()[1]
        u_new = torch.zeros(3, n // 2, n // 2, n // 2, device=u.device, dtype=u.dtype)

        # Verwende ganzzahlige Divisionen (//) für alle Indizes
        u_new[:, :n // 4, :n // 4, :n // 4] = u[:, :n // 4, :n // 4, :n // 4]

        u_new[0, :n // 4, n // 4:, :n // 4] = torch.flip(torch.transpose(u[1, :n // 4, :n // 4, :n // 4], 0, 1), [1])
        u_new[1, :n // 4, n // 4:, :n // 4] = -1 * torch.flip(torch.transpose(u[0, :n // 4, :n // 4, :n // 4], 0, 1), [1])
        u_new[2, :n // 4, n // 4:, :n // 4] = torch.flip(torch.transpose(u[2, :n // 4, :n // 4, :n // 4], 0, 1), [1])

        u_new[0, n // 4:, :, :n // 4] = -1 * torch.flip(torch.flip(u_new[0, :n // 4, :, :n // 4], [0]), [1])
        u_new[1, n // 4:, :, :n // 4] = -1 * torch.flip(torch.flip(u_new[1, :n // 4, :, :n // 4], [0]), [1])
        u_new[2, n // 4:, :, :n // 4] = torch.flip(torch.flip(u_new[2, :n // 4, :, :n // 4], [0]), [1])

        u_new[0, :, :, n // 4:] = torch.flip(u_new[0, :, :, :n // 4], [2])
        u_new[1, :, :, n // 4:] = torch.flip(u_new[1, :, :, :n // 4], [2])
        u_new[2, :, :, n // 4:] = -1 * torch.flip(u_new[2, :, :, :n // 4], [2])

        Symmetrie = torch.zeros(8, device=u.device, dtype=u.dtype)
        Symmetrie_mean = torch.zeros(8, device=u.device, dtype=u.dtype)
        top_symmetry_coords = {}  # Dictionary für die Top-1%-Koordinaten pro Symmetrie-Bereich

        # Symmetrie-Berechnungen
        Symmetrie[0] = torch.max(torch.norm(u[:, :n // 2, :n // 2, :n // 2] - u_new, dim=0))
        Symmetrie_mean[0] = torch.mean(torch.norm(u[:, :n // 2, :n // 2, :n // 2] - u_new, dim=0))

        Symmetrie[1] = torch.max(torch.norm(u[:, n // 2:, n // 2:, :n // 2] - u_new, dim=0))
        Symmetrie_mean[1] = torch.mean(torch.norm(u[:, n // 2:, n // 2:, :n // 2] - u_new, dim=0))

        Symmetrie[2] = torch.max(torch.norm(u[:, n // 2:, :n // 2, n // 2:] - u_new, dim=0))
        Symmetrie_mean[2] = torch.mean(torch.norm(u[:, n // 2:, :n // 2, n // 2:] - u_new, dim=0))

        Symmetrie[3] = torch.max(torch.norm(u[:, :n // 2, n // 2:, n // 2:] - u_new, dim=0))
        Symmetrie_mean[3] = torch.mean(torch.norm(u[:, :n // 2, n // 2:, n // 2:] - u_new, dim=0))

        u_new2 = torch.flip(u_new, [1])
        u_new2[0, :, :, :] = -1 * u_new2[0, :, :, :]

        Symmetrie[4] = torch.max(torch.norm(u[:, :n // 2, :n // 2, n // 2:] - u_new2, dim=0))
        Symmetrie_mean[4] = torch.mean(torch.norm(u[:, :n // 2, :n // 2, n // 2:] - u_new2, dim=0))

        Symmetrie[5] = torch.max(torch.norm(u[:, n // 2:, n // 2:, n // 2:] - u_new2, dim=0))
        Symmetrie_mean[5] = torch.mean(torch.norm(u[:, n // 2:, n // 2:, n // 2:] - u_new2, dim=0))

        Symmetrie[6] = torch.max(torch.norm(u[:, n // 2:, :n // 2, :n // 2] - u_new2, dim=0))
        Symmetrie_mean[6] = torch.mean(torch.norm(u[:, n // 2:, :n // 2, :n // 2] - u_new2, dim=0))

        Symmetrie[7] = torch.max(torch.norm(u[:, :n // 2, n // 2:, :n // 2] - u_new2, dim=0))
        Symmetrie_mean[7] = torch.mean(torch.norm(u[:, :n // 2, n // 2:, :n // 2] - u_new2, dim=0))

        Symmetrie_max = torch.max(Symmetrie) / torch.max(torch.norm(u, dim=0))
        Symmetrie_mean_final = torch.mean(Symmetrie_mean) / torch.mean(torch.abs(torch.norm(u, dim=0)))

        return torch.stack([Symmetrie_max, Symmetrie_mean_final])


class SymmetryTopPercentageReporter(Observable):

    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        u = self.lattice.u(f)

        n = u.size()[1]
        u_new = torch.zeros(3, n // 2, n // 2, n // 2, device=u.device, dtype=u.dtype)

        # Verwende ganzzahlige Divisionen (//) für alle Indizes
        u_new[:, :n // 4, :n // 4, :n // 4] = u[:, :n // 4, :n // 4, :n // 4]

        u_new[0, :n // 4, n // 4:, :n // 4] = torch.flip(torch.transpose(u[1, :n // 4, :n // 4, :n // 4], 0, 1), [1])
        u_new[1, :n // 4, n // 4:, :n // 4] = -1 * torch.flip(torch.transpose(u[0, :n // 4, :n // 4, :n // 4], 0, 1), [1])
        u_new[2, :n // 4, n // 4:, :n // 4] = torch.flip(torch.transpose(u[2, :n // 4, :n // 4, :n // 4], 0, 1), [1])

        u_new[0, n // 4:, :, :n // 4] = -1 * torch.flip(torch.flip(u_new[0, :n // 4, :, :n // 4], [0]), [1])
        u_new[1, n // 4:, :, :n // 4] = -1 * torch.flip(torch.flip(u_new[1, :n // 4, :, :n // 4], [0]), [1])
        u_new[2, n // 4:, :, :n // 4] = torch.flip(torch.flip(u_new[2, :n // 4, :, :n // 4], [0]), [1])

        u_new[0, :, :, n // 4:] = torch.flip(u_new[0, :, :, :n // 4], [2])
        u_new[1, :, :, n // 4:] = torch.flip(u_new[1, :, :, :n // 4], [2])
        u_new[2, :, :, n // 4:] = -1 * torch.flip(u_new[2, :, :, :n // 4], [2])

        u_new2 = torch.flip(u_new, [1])
        u_new2[0, :, :, :] = -1 * u_new2[0, :, :, :]

        top_symmetry_coords = {}

        regions = [
            (u[:, :n // 2, :n // 2, :n // 2], u_new),
            (u[:, n // 2:, n // 2:, :n // 2], u_new),
            (u[:, n // 2:, :n // 2, n // 2:], u_new),
            (u[:, :n // 2, n // 2:, n // 2:], u_new),
            (u[:, :n // 2, :n // 2, n // 2:], u_new2),
            (u[:, n // 2:, n // 2:, n // 2:], u_new2),
            (u[:, n // 2:, :n // 2, :n // 2], u_new2),
            (u[:, :n // 2, n // 2:, :n // 2], u_new2)
        ]

        for i, (region, reference) in enumerate(regions):
            diff = torch.norm(region - reference, dim=0)
            num_top = max(1, int(diff.numel() * 0.01))

            top_values, top_indices = torch.topk(diff.flatten(), num_top)

            n1, n2, n3 = diff.shape
            top_x = torch.div(top_indices, (n2 * n3), rounding_mode='floor')
            top_y = torch.div(torch.remainder(top_indices, (n2 * n3)), n3, rounding_mode='floor')
            top_z = torch.remainder(top_indices, n3)

            top_symmetry_coords[i] = torch.stack((top_x, top_y, top_z), dim=1)

        # ✅ Fix: Dictionary in einen Tensor umwandeln
        all_coords = torch.cat([coords.flatten() for coords in top_symmetry_coords.values()])

        return all_coords  # Jetzt gibt die Methode einen Tensor zurück

class WallQuantities(Observable):
    def __init__(self, lattice, flow, averaging_steps=100, distance_to_wall=1.0,
                 smagorinsky_constant=0.17, delta_x=1.0, normal_axis=1, wall: str = 'bottom'):
        super().__init__(lattice, flow)
        self.rho_lu = 1.0
        self.molecular_nu_lu = flow.units.viscosity_lu
        self.half_channel_height_lu = flow.resolution_y / 2

        self.smagorinsky_constant = smagorinsky_constant
        self.delta_x = delta_x
        self.normal_axis = normal_axis
        self.wall = wall

        if self.wall == 'bottom':
            self.fluid_layer_idx = 1
            self.wall_layer_idx = 0
        elif self.wall == 'top':
            self.fluid_layer_idx = flow.resolution_y - 2
            self.wall_layer_idx = flow.resolution_y - 1
        else:
            raise ValueError(f"Unsupported wall type: {self.wall}. Must be 'bottom' or 'top'.")

        self.distance_to_wall = distance_to_wall
        self.averaging_steps = averaging_steps
        self.current_step = 0

        self.u_tau_history = []
        self.re_tau_history = []
        self.y_plus_history = []

        self.ndim = len(flow.grid)

        # Init last u_tau mean to 0.0
        self.last_u_tau_spatial_mean = 0.0

    def __call__(self, f):
        u = self.lattice.u(f)
        rho = self.lattice.rho(f).squeeze()
        u_x = u[0].squeeze()
        u_z = u[2].squeeze() if self.ndim == 3 else torch.zeros_like(u_x)

        dims = u_x.ndim
        fluid_slice_indices = [slice(None)] * dims
        fluid_slice_indices[self.normal_axis] = self.fluid_layer_idx

        mask = torch.zeros_like(u_x, dtype=torch.bool, device=u_x.device)
        mask[tuple(fluid_slice_indices)] = True

        rho_f = rho[mask]
        u_x_f = u_x[mask]
        u_z_f = u_z[mask]

        sign = 1.0 if self.wall == 'bottom' else -1.0
        du_dn = sign * u_x_f / self.distance_to_wall

        magnitude_of_gradient = torch.abs(du_dn)
        nu_turbulent = (self.smagorinsky_constant * self.delta_x) ** 2 * magnitude_of_gradient
        nu_effective = torch.clamp(self.molecular_nu_lu + nu_turbulent, min=self.molecular_nu_lu)

        u_tau_from_gradient = torch.sqrt(torch.abs(nu_effective * du_dn))
        tau_w_magnitude = rho_f * u_tau_from_gradient ** 2

        u_mag_f = torch.sqrt(u_x_f ** 2 + u_z_f ** 2)
        dir_x = torch.where(u_mag_f > 1e-10, u_x_f / u_mag_f, torch.tensor(0.0, device=u_mag_f.device))
        dir_z = torch.where(u_mag_f > 1e-10, u_z_f / u_mag_f, torch.tensor(0.0, device=u_mag_f.device))

        tau_w_x = - dir_x * tau_w_magnitude
        tau_w_z = - dir_z * tau_w_magnitude

        if self.ndim == 2:
            tau_mag = tau_w_x
        elif self.ndim == 3:
            tau_mag = torch.sqrt(tau_w_x ** 2 + tau_w_z ** 2)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.ndim}")

        u_tau_current = torch.sqrt(torch.abs(tau_mag) / self.rho_lu)

        mean_nu_eff = torch.mean(nu_effective).item()
        mean_nu_eff = max(mean_nu_eff, 1e-10)

        re_tau = u_tau_current * self.half_channel_height_lu / mean_nu_eff
        y_plus = self.distance_to_wall * u_tau_current / mean_nu_eff

        self.u_tau_history.append(torch.mean(u_tau_current).item())
        self.re_tau_history.append(torch.mean(re_tau).item())
        self.y_plus_history.append(torch.mean(y_plus).item())

        self.current_step += 1

        # Speichere IMMER den letzten Wert – für AdaptiveForce wichtig
        self.last_u_tau_spatial_mean = torch.mean(u_tau_current).item()

        if self.current_step >= self.averaging_steps:
            avg_u_tau = np.mean(self.u_tau_history)
            avg_re_tau = np.mean(self.re_tau_history)
            avg_y_plus = np.mean(self.y_plus_history)

            print(f"[WallQuantities - {self.wall}] Avg Re_tau: {avg_re_tau:.2f}, y+_avg: {avg_y_plus:.2f}")

            self.u_tau_history.clear()
            self.re_tau_history.clear()
            self.y_plus_history.clear()
            self.current_step = 0

            return torch.tensor([avg_re_tau, avg_y_plus], device=self.lattice.device)
        else:
            return torch.zeros(2, device=self.lattice.device)

    def utau(self):
        """
        Gibt den zuletzt gemittelten u_tau-Wert zurück.
        """
        return self.last_u_tau_spatial_mean

class GlobalMeanUXReporter(Observable):
    def __init__(self, lattice, flow):
        super().__init__(lattice, flow)
        self.current_mean_ux_lu = torch.tensor(0.0, device=self.lattice.device)

    def __call__(self, f):
        u_field_lu = self.lattice.u(f)  # u_field_lu: shape (3, Nx, Ny, Nz)
        u_x_spatial = u_field_lu[0]     # Nur die x-Komponente
        self.current_mean_ux_lu = torch.mean(u_x_spatial)
        return self.current_mean_ux_lu

    def value(self):
        return self.current_mean_ux_lu
