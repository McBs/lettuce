import numpy as np
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet


class ChannelFlow2D(object):
    """
    Flow class to simulate the flow around an object (mask) in 2D.
    It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    Parameters
    ----------
    resolution_x : int
        Grid resolution in streamwise direction.
    resolution_y : int
        Grid resolution in spanwise direction.
    char_length_lu : float
        The characteristic length in lattice units; usually the number of grid points for the obstacle in flow direction

    Attributes
    ----------
    mask : np.array with dtype = bool
        Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------
    Initialization of flow around a cylinder:

    >>> from lettuce import Lattice, D2Q9
    >>> flow = Obstacle2D(
    >>>     resolution_x=101,
    >>>     resolution_y=51,
    >>>     reynolds_number=100,
    >>>     mach_number=0.1,
    >>>     lattice=lattice,
    >>>     char_length_lu=10
    >>> )
    >>> x, y = flow.grid
    >>> x = flow.units.convert_length_to_lu(x)
    >>> y = flow.units.convert_length_to_lu(y)
    >>> condition = np.sqrt((x-25)**2+(y-25)**2) < 5.0001
    >>> flow.mask[np.where(condition)] = 1
   """

    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y), dtype=bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y)
        self._mask = m.astype(bool)

    def initial_solution(self, grid):
        xg, yg = grid

        p = np.ones_like(xg)[None, ...]

        # Auflösung
        ndir = 2
        nx, ny = self.resolution_x, self.resolution_y
        shape = (ndir, nx, ny)
        shape_hat = (ndir, nx, ny // 2)

        # Vektorpotential ψ ∈ [-1, 1]
        # Vertikale Gewichtung (1 an Wand, 0 in der Mitte)
        y = np.linspace(0, 1, ny)
        weight_y = 1.0 - np.exp(-((y - 0.0) / 0.2) ** 2) - np.exp(-((y - 1.0) / 0.2) ** 2)
        weight_y = 1.0 - weight_y / weight_y.max()  # Skaliert zwischen 0 (Mitte) und 1 (Wand)

        # 2D Gewichtungsmaske (broadcastfähig)
        weight_2d = weight_y[None, :]  # shape = [1, ny]

        # Auf jedes Richtungsfeld anwenden
        random_psi = ((np.random.random(shape) - 0.5) * 2) * weight_2d

        # Filterparameter
        k0 = np.sqrt(nx ** 2 + ny ** 2)
        psi_hat_scaled = np.empty(shape_hat, dtype=complex)
        psi_scaled = np.empty(shape)
        for d in range(ndir):
            psi_hat = np.fft.fftn(random_psi[d], s=shape_hat[1:], axes=[0, 1])
            kxs, kys = np.meshgrid(np.arange(shape_hat[2]), np.arange(shape_hat[1]), indexing='xy')
            kabs = np.sqrt(kxs ** 2 + kys ** 2)
            psi_hat *= np.exp(-2 * kabs / k0)
            psi_hat[0, 0] = 0  # k=0 Mode entfernen
            psi_hat_scaled[d] = psi_hat
            psi_scaled[d] = np.real(np.fft.ifftn(psi_hat_scaled[d], s=shape[1:]))

        # Gradient von ψ
        gradOf_psi_scaled = np.array([np.gradient(psi_scaled[d]) for d in range(ndir)])  # shape = [2][2][nx][ny]

        # 2D Curl:
        # u_x = dψ_y/dy - dψ_x/dx (→ in 2D reduziert sich das auf Skalarform)
        u = np.zeros((2, nx, ny), dtype=float)
        u[0] = gradOf_psi_scaled[1][0] - gradOf_psi_scaled[0][1]  # u_x = dψ_y/dx - dψ_x/dy
        u[1] = gradOf_psi_scaled[0][0] + gradOf_psi_scaled[1][1] * 0  # optional null oder andere Komponente

        # Optional: Maske anwenden
        u *= (1 - self.mask.astype(float))

        # Optional: Maximalgeschwindigkeit normieren
        target_umax = 1
        current_umax = np.max(np.sqrt(np.sum(u ** 2, axis=0)))
        if current_umax > 0:
            u *= target_umax / current_umax
        # Basisströmung: Poiseuille in x-Richtung
        channel_height_lu = self.resolution_y / self.units.characteristic_length_lu
        y_normalized = yg / channel_height_lu
        base_umax = 1
        u_base = base_umax * y_normalized * (1 - y_normalized)
        u[0] += u_base * (1 - self.mask.astype(float))
        return p, u

    @property
    def grid(self):
        stop_x = self.resolution_x / self.units.characteristic_length_lu
        stop_y = self.resolution_y / self.units.characteristic_length_lu

        x = np.linspace(0, stop_x, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, stop_y, num=self.resolution_y, endpoint=False)

        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        return [

            BounceBackBoundary(self.mask, self.units.lattice)
        ]


class ChannelFlow3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=1,
            characteristic_velocity_pu=1)

        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=bool)

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(bool)

    def initial_solution(self, grid):
        xg, yg, zg = grid

        # Druckfeld: konstant
        p = np.ones_like(xg)[None, ...]

        # Auflösung
        ndir = 3
        nx, ny, nz = self.resolution_x, self.resolution_y, self.resolution_z
        shape = (ndir, nx, ny, nz)
        shape_hat = (ndir, nx, ny, nz // 2 + 1)  # korrekt für rfft

        # Vektorpotential ψ ∈ [-1, 1]
        # Turbulenzgewichtung in y-Richtung (wandnah = 1, mitte = 0)
        y = np.linspace(0, 1, ny)
        weight_y = np.exp(-((y - 0.0) / 0.2) ** 2) + np.exp(-((y - 1.0) / 0.2) ** 2)
        weight_y /= weight_y.max()  # normiert auf [0,1]

        # Auf 3D-Maske erweitern
        weight_3d = weight_y[None, :, None]  # shape = [1, ny, 1], broadcastbar auf (nx, ny, nz)

        # Gewichtetes ψ-Feld
        random_psi = ((np.random.rand(*shape) - 0.5) * 2) * weight_3d

        # FFT und Filterung
        k0 = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        psi_hat_scaled = np.empty(shape_hat, dtype=complex)
        psi_scaled = np.empty(shape)
        for d in range(ndir):
            # FFT in 3D
            psi_hat = np.fft.rfftn(random_psi[d], s=shape[1:], axes=(0, 1, 2))

            # Frequenzgitter passend zu rfft
            kx = np.fft.fftfreq(nx)
            ky = np.fft.fftfreq(ny)
            kz = np.fft.rfftfreq(nz)  # da rfft
            kxs, kys, kzs = np.meshgrid(kx, ky, kz, indexing='ij')
            kabs = np.sqrt(kxs ** 2 + kys ** 2 + kzs ** 2)

            # Filter
            psi_hat *= np.exp(-2 * kabs * nx / k0)
            psi_hat[0, 0, 0] = 0  # DC-Mode entfernen
            psi_hat_scaled[d] = psi_hat

            # IFFT zurück in realen Raum
            psi_scaled[d] = np.fft.irfftn(psi_hat_scaled[d], s=shape[1:], axes=(0, 1, 2))

        # Gradient von ψ berechnen → shape: [3][3][nx][ny][nz]
        gradOf_psi_scaled = np.array([np.gradient(psi_scaled[d]) for d in range(ndir)])

        # Curl berechnen
        u = np.zeros(shape)
        u[0] = gradOf_psi_scaled[2][1] - gradOf_psi_scaled[1][2]
        u[1] = gradOf_psi_scaled[0][2] - gradOf_psi_scaled[2][0]
        u[2] = gradOf_psi_scaled[1][0] - gradOf_psi_scaled[0][1]

        # Maske anwenden (z. B. Hindernis)
        u *= (1 - self.mask.astype(float))

        # Optional: Skalierung auf gewünschte Maximalgeschwindigkeit
        target_umax = 1
        current_umax = np.max(np.sqrt(np.sum(u ** 2, axis=0)))
        if current_umax > 0:
            u *= target_umax / current_umax
        # Basisgeschwindigkeit: parabolisches Profil in x-Richtung
        # Kanalhöhe in LU
        # Basisgeschwindigkeit: parabolisches Profil in x-Richtung (nur y-abhängig!)
        channel_height_lu_y = self.resolution_y / self.units.characteristic_length_lu
        y_normalized = yg / channel_height_lu_y

        base_umax = 1  # sinnvoller Maximalwert in LU
        u_base = base_umax * y_normalized * (1 - y_normalized)

        u[0] += u_base * (1 - self.mask.astype(float))

        return p, u

    @property
    def grid(self):
        stop_x = self.resolution_x / self.units.characteristic_length_lu
        stop_y = self.resolution_y / self.units.characteristic_length_lu
        stop_z = self.resolution_z / self.units.characteristic_length_lu

        x = np.linspace(0, stop_x, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, stop_y, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, stop_z, num=self.resolution_z, endpoint=False)

        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid
        return [
                BounceBackBoundary(self.mask, self.units.lattice)]

