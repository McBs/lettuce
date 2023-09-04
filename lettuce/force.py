import torch
import numpy as np

__all__ = ["Guo", "ShanChen", "ForceSpectral", "TrigonometicForce", "LinearForce"]


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


class ForceSpectral:
    """
    Description
    """

    def __init__(self, lattice, flow, power=1, ka=None, kb=None, c=None, dt=1):
        self.device = lattice.device
        self.dtype = lattice.dtype
        self.ka = torch.tensor(ka, dtype=self.dtype, device=self.device)
        self.kb = torch.tensor(kb, dtype=self.dtype, device=self.device)
        self.kf = (self.ka + self.kb) / 2
        self.c = torch.tensor(c, dtype=self.dtype, device=self.device)
        self.power = torch.tensor(flow.units.convert_powerforce_to_lu(2 * power), dtype=self.dtype, device=self.device)
        self.dt = torch.tensor(dt, dtype=self.dtype, device=self.device)
        self.resolution = flow.resolution

        frequencies = [np.fft.fftfreq(dim, d=1 / dim) for dim in [self.resolution] * 3]
        k = np.array(np.meshgrid(*frequencies))
        k[0][k[0] == 0] = 1e-15
        k[1][k[1] == 0] = 1e-15
        k[2][k[2] == 0] = 1e-15
        kk = np.linalg.norm(k, axis=0)
        k = torch.tensor(np.array(k), device=self.device, dtype=self.dtype)
        self.kk = torch.tensor(np.array(kk), device=self.device, dtype=self.dtype)
        self.k = k
        self.e1 = [k[1] * (k[0] ** 2 + k[1] ** 2) ** (-0.5),
                   -k[0] * (k[0] ** 2 + k[1] ** 2) ** (-0.5),
                   torch.zeros_like(k[0])]
        self.e2 = [k[0] * k[2] * (self.kk * torch.sqrt(k[0] ** 2 + k[1] ** 2)) ** -1,
                   k[1] * k[2] * (self.kk * torch.sqrt(k[0] ** 2 + k[1] ** 2)) ** -1,
                   - torch.sqrt(k[0] ** 2 + k[1] ** 2) * self.kk ** -1]

        self.Fk_norm = self._normalization()

        one = torch.tensor(1, dtype=self.dtype, device=self.device)
        assert (torch.isclose(torch.sqrt(self.e1[0] ** 2 + self.e1[1] ** 2 + self.e1[2] ** 2).max(), one))
        assert (torch.isclose(torch.sqrt(self.e1[0] ** 2 + self.e1[1] ** 2 + self.e1[2] ** 2).min(), one))
        assert (torch.isclose(torch.sqrt(self.e2[0] ** 2 + self.e2[1] ** 2 + self.e2[2] ** 2).max(), one))
        assert (torch.isclose(torch.sqrt(self.e2[0] ** 2 + self.e2[1] ** 2 + self.e2[2] ** 2).min(), one))

        F, Fh = self.__call__(fh=True)
        FhFh = torch.stack([(Fh[_] * torch.conj(Fh[_])) for _ in range(3)]).sum(0)
        dk = 0.5
        average, wavenumbers = self._spectral_average(FhFh, self.kk, dk=dk)
        P_h = (2 * torch.pi * wavenumbers ** 2 * average * dk).sum().item()
        P_pu = 0.5 * torch.stack([((F[_].real) ** 2 + (F[_].imag) ** 2) for _ in range(3)]).sum(0).mean().item()
        # assert (torch.isclose(self.power, torch.tensor(P_h, dtype=self.dtype, device=self.device)))
        # assert (torch.isclose(self.power, torch.tensor(P_pu, dtype=self.dtype, device=self.device)))

    def __call__(self, fh=False):
        Fh = self.spectral_force()
        F = torch.stack([(torch.fft.ifftn(Fh[i], dim=tuple(torch.arange(3)), norm="forward")) for i in range(3)])
        if fh:
            return F, Fh
        else:
            return F.real

    def spectral_force(self, ):
        RealRandom, ImagRandom = self.get_complex_factors(self.resolution)

        Fk = torch.exp(-(self.kk - self.kf) ** 2 * self.c ** -1)
        Fk.ravel()[0] = 0
        Fk *= (self.Fk_norm) ** -1 * self.power * self.dt ** -1
        Fk = torch.sqrt(Fk * (2 * torch.pi * self.kk ** 2) ** -1)

        A = Fk * torch.complex(RealRandom[0], ImagRandom[0])
        B = Fk * torch.complex(RealRandom[1], ImagRandom[1])
        Fh = torch.stack([A * self.e1[_] + B * self.e2[_] for _ in range(3)])

        Fh[0].ravel()[0] = 0
        Fh[1].ravel()[0] = 0
        Fh[2].ravel()[0] = 0
        return Fh

    def get_complex_factors(self, k):
        phi = torch.rand([k] * 3, dtype=self.dtype, device=self.device) * 2 * torch.pi
        theta = torch.rand([2] + [k] * 3, dtype=self.dtype, device=self.device) * 2 * torch.pi
        ga = torch.sin(phi)
        gb = torch.cos(phi)

        RealRandom = [torch.cos(theta[_]) * g for _, g in enumerate([ga, gb])]
        ImagRandom = [torch.sin(theta[_]) * g for _, g in enumerate([ga, gb])]

        return RealRandom, ImagRandom

    def _get_random_factors(self, wavenumbers):
        return torch.rand(wavenumbers, dtype=self.dtype, device=self.device)

    def _normalization(self):
        Ka = -0.5 * torch.sqrt(torch.tensor(torch.pi)) * torch.sqrt(self.c) * torch.erf(
            (self.kf - self.ka) / torch.sqrt(self.c))
        Kb = -0.5 * torch.sqrt(torch.tensor(torch.pi)) * torch.sqrt(self.c) * torch.erf(
            (self.kf - self.kb) / torch.sqrt(self.c))
        return Kb - Ka

    @staticmethod
    def _spectral_average(f, wavenorms, dk=1):
        k = [1];
        eps = 1e-10
        while k[-1] < int(f.shape[0] / 2) - eps:
            k.append(k[-1] + dk)
        wavenumbers = torch.tensor(k)
        average = torch.zeros_like(wavenumbers, dtype=torch.float32)

        for idnr, k in enumerate(wavenumbers[1:-1]):
            condition = (wavenorms <= (float(k) + dk)) & (wavenorms > (float(k) - dk))
            counter = condition.sum()
            average[idnr + 1] = f[condition].sum() / counter

        condition = (wavenorms <= (wavenumbers[0] + dk))
        counter = condition.sum()
        average[0] = f[condition].sum() / counter

        condition = (wavenorms <= wavenumbers[-1]) & (wavenorms > (wavenumbers[-1] - dk))
        counter = condition.sum()
        average[-1] = f[condition].sum() / counter
        return average, wavenumbers


class TrigonometicForce:
    """
    Description
    """

    def __init__(self, lattice, flow, power=1, Lc=2 * torch.pi, ka=1, kb=2):
        self.lattice = lattice
        self.grid = torch.tensor(flow.grid, device=lattice.device, dtype=lattice.dtype)
        self.Lc = Lc
        self.ka = ka
        self.kb = kb
        self.power = torch.tensor(flow.units.convert_powerforce_to_lu(power), device=lattice.device, dtype=lattice.dtype)
        self.phase = torch.randn(6, device=self.lattice.device, dtype=self.lattice.dtype)
        # self.power = torch.tensor(power, device=lattice.device, dtype=lattice.dtype)
        print("Initialize Trigonometric excitation")

    def __call__(self):
        self.phase += torch.randn(6, device=self.lattice.device, dtype=self.lattice.dtype)*0.1
        F = (torch.stack([(
                (torch.stack([(
                    torch.stack([(
                        self._C(j, d) * torch.sin(2 * torch.pi * xx / self.Lc * k + self.phase[j]) +
                        self._C(j, d) * torch.cos(2 * torch.pi * xx / self.Lc * k + self.phase[3+j])
                    ) / 3 for j, xx in enumerate(self.grid)], 0).sum(0)
                ) / torch.sqrt(torch.tensor(self.kb - self.ka + 1)) for k in range(self.ka, self.kb + 1)],0).sum(0))
        ) for d in range(self.lattice.D)], 0))
        return F

    def _phase(self):
        return torch.randn(1, device=self.lattice.device, dtype=self.lattice.dtype)
    #         return 0

    def _C(self, j, d):
        return torch.randn(1, device=self.lattice.device, dtype=self.lattice.dtype) * torch.sqrt(2 * self.power)

class LinearForce:
    """
    Description
    """

    def __init__(self, lattice, flow, power=0.01):
        self.lattice = lattice
        self.grid = torch.tensor(flow.grid, device=lattice.device, dtype=lattice.dtype)
        self.power = torch.tensor(flow.units.convert_powerforce_to_lu(power), device=lattice.device, dtype=lattice.dtype)
        print("Initialize linear excitation")

    def __call__(self, f):
        u = self.lattice.u(f)
        u -= (u.mean([1, 2, 3])[:, None, None, None])
        uu_rms = self.lattice.einsum("i,i->", [u, u]).mean()
        return self.power / uu_rms * u