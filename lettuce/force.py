__all__ = ["Guo", "ShanChen", "AdaptiveForce"]

import torch
class Guo:
    def __init__(self, lattice, tau, acceleration):
        self.lattice = lattice
        self.tau = tau
        self.acceleration = lattice.convert_to_tensor(acceleration)

    def source_term(self, u):
        index = [Ellipsis] + [None] * self.lattice.D
        emu = self.lattice.e[index] - u
        eu = self.lattice.einsum("ib,b->i", [self.lattice.e, u])
        eeu = self.lattice.einsum("ia,i->ia", [self.lattice.e, eu])
        emu_eeu = emu / (self.lattice.cs ** 2) + eeu / (self.lattice.cs ** 4)
        emu_eeuF = self.lattice.einsum("ia,a->i", [emu_eeu, self.acceleration])
        weemu_eeuF = self.lattice.w[index] * emu_eeuF
        return (1 - 1 / (2 * self.tau)) * weemu_eeuF

    def u_eq(self, f):
        index = [Ellipsis] + [None] * self.lattice.D
        return self.ueq_scaling_factor * self.acceleration[index] / self.lattice.rho(f)

    @property
    def ueq_scaling_factor(self):
        return 0.5


class ShanChen:
    def __init__(self, lattice, tau, acceleration):
        self.lattice = lattice
        self.tau = tau
        self.acceleration = lattice.convert_to_tensor(acceleration)

    def source_term(self, u):
        return 0

    def u_eq(self, f):
        index = [Ellipsis] + [None] * self.lattice.D
        return self.ueq_scaling_factor * self.acceleration[index] / self.lattice.rho(f)

    @property
    def ueq_scaling_factor(self):
        return self.tau * 1


import numpy as np

class AdaptiveForce:
    def __init__(self, lattice, flow, target_u_m_lu,
                 wall_bottom,
                 wall_top,
                 global_ux_reporter,
                 base_lbm_tau_lu):

        self.lattice = lattice
        self.u_m = target_u_m_lu
        self.wall_bottom = wall_bottom
        self.wall_top = wall_top
        self.global_ux = global_ux_reporter
        self.H = flow.resolution_y / 2.0  # Halbkanalhöhe in LU
        self.base_lbm_tau = base_lbm_tau_lu
        self.ueq_scaling_factor = 0.5
        self.last_force_lu = lattice.convert_to_tensor([0.0] * lattice.D)

    def compute_force(self):
        utau_b_lu = self.wall_bottom.u_tau_mean
        utau_t_lu = self.wall_top.u_tau_mean

        # 🔐 Schutz gegen None
        if utau_b_lu is None or utau_t_lu is None:
            print("⚠️ u_tau_mean nicht verfügbar. AdaptiveForce überspringt Kraftberechnung.")
            return self.last_force_lu  # oder return torch.zeros_like(self.last_force_lu)

        utau_mean_lu = 0.5 * (utau_b_lu + utau_t_lu)
        ux_mean_lu = self.global_ux.value()

        Fx_lu = (utau_mean_lu ** 2) / self.H + (self.u_m - ux_mean_lu) * (self.u_m / self.H)

        if torch.isnan(Fx_lu) or torch.isinf(Fx_lu):
            print("⚠️ Warnung: Fx_lu enthält NaN oder Inf.")
            print(
                f"Terme: utau²/H = {(utau_mean_lu ** 2) / self.H}, Korrekturterm = {(self.u_m - ux_mean_lu) * (self.u_m / self.H)}")

        self.last_force_lu = self.lattice.convert_to_tensor([Fx_lu] + [0.0] * (self.lattice.D - 1))
        return self.last_force_lu

    def __call__(self, u_field_lu, f):
        """
        Wird von SmagorinskyCollision (als Forcing) aufgerufen.
        """
        # Aktualisiere Kraft
        self.compute_force()

        # Berechne Guo-Source-Term auf Basis der aktuellen Kraft
        guo_force = Guo(self.lattice, tau=self.base_lbm_tau, acceleration=self.last_force_lu)
        return guo_force.source_term(u_field_lu)

    def u_eq(self, f):
        rho = self.lattice.rho(f)

        index = [Ellipsis] + [None] * self.lattice.D
        # Stellen Sie sicher, dass keine Division durch Null durch rho erfolgt
        denom = torch.where(rho < 1e-10, torch.tensor(1e-10, device=rho.device, dtype=rho.dtype), rho)

        return self.ueq_scaling_factor * self.last_force_lu[index] / denom

