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

        # --- 🌀 Vektorpotential ψ ∈ [-1, 1] ---
        # Normiertes y ∈ [0,1]
        y_normalized = yg / yg.max()

        # Vertikale Gewichtung: hohe Energie an den Wänden
        weight_y = np.exp(-((y_normalized - 0.0) / 0.2) ** 2) + np.exp(-((y_normalized - 1.0) / 0.2) ** 2)
        weight_y /= weight_y.max()
        weight_2d = weight_y  # shape [nx, ny] → wie yg

        # Zufälliges Feld für ψ, skaliert mit Wandgewichtung
        random_psi = ((np.random.rand(*shape) - 0.5) * 2) * weight_2d[None, :, :]

        # --- 🎚️ Glätten mit FFT Lowpass-Filter ---
        k0 = np.sqrt(nx ** 2 + ny ** 2)
        psi_filtered = np.empty_like(random_psi)
        for d in range(ndir):
            psi_hat = np.fft.fft2(random_psi[d])
            kx = np.fft.fftfreq(nx).reshape(-1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1)
            kabs = np.sqrt((kx * nx) ** 2 + (ky * ny) ** 2)
            filter_mask = np.exp(-2 * kabs / k0)
            psi_hat *= filter_mask
            psi_hat[0, 0] = 0  # DC-Komponente entfernen
            psi_filtered[d] = np.real(np.fft.ifft2(psi_hat))

        # --- 🌀 Geschwindigkeit aus Curl(ψ) ---
        u = np.zeros_like(psi_filtered)
        u[0] = np.gradient(psi_filtered[1], axis=0) - np.gradient(psi_filtered[0], axis=1)  # u_x = dψ_y/dx - dψ_x/dy
        u[1] = np.zeros_like(u[0])  # optional für u_y = 0

        # --- 🎯 Normierung der Störung ---
        target_umax = 1  # z.B. 10 % der Basisgeschwindigkeit
        umax = np.max(np.sqrt(np.sum(u ** 2, axis=0)))
        if umax > 0:
            u *= target_umax / umax

        # --- ➕ Poiseuille-Profil (Basisströmung in x) ---
        y_normalized = yg / yg.max()
        base_umax = 1.0  # maximale Basisgeschwindigkeit
        u_base = base_umax * y_normalized * (1 - y_normalized)
        u[0] += u_base * (1 - self.mask.astype(float))

        return p, u

    @property
    def grid(self):
        stop_x = self.resolution_x / self.units.characteristic_length_lu
        stop_y = self.resolution_y / self.units.characteristic_length_lu

        x = np.linspace(0, stop_x, num=self.resolution_x, endpoint=True)
        y = np.linspace(0, stop_y, num=self.resolution_y, endpoint=True)

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
        p = np.ones_like(xg)[None, ...]

        # Auflösung
        ndir = 3
        nx, ny, nz = self.resolution_x, self.resolution_y, self.resolution_z
        shape = (ndir, nx, ny, nz)

        # --- 🌀 Vektorpotential ψ ∈ [-1, 1] ---
        y_normalized = yg / yg.max()

        # Vertikale Gewichtung für Wände
        weight_y = np.exp(-((y_normalized - 0.0) / 0.2) ** 2) + np.exp(-((y_normalized - 1.0) / 0.2) ** 2)
        weight_y /= weight_y.max()
        weight_3d = weight_y  # shape (nx, ny, nz)

        # Zufälliges ψ-Feld, skaliert mit Gewichtung
        random_psi = ((np.random.rand(*shape) - 0.5) * 2) * weight_3d[None, :, :, :]

        # --- 🎚️ FFT Lowpass-Filter ---
        k0 = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        psi_filtered = np.empty_like(random_psi)
        for d in range(ndir):
            psi_hat = np.fft.fftn(random_psi[d])
            kx = np.fft.fftfreq(nx).reshape(-1, 1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1, 1)
            kz = np.fft.fftfreq(nz).reshape(1, 1, -1)
            kabs = np.sqrt((kx * nx) ** 2 + (ky * ny) ** 2 + (kz * nz) ** 2)
            filter_mask = np.exp(-2 * kabs / k0)
            psi_hat *= filter_mask
            psi_hat[0, 0, 0] = 0  # DC entfernen
            psi_filtered[d] = np.real(np.fft.ifftn(psi_hat))

        # --- 🌀 Curl(ψ) ergibt Geschwindigkeit ---
        u = np.zeros_like(psi_filtered)
        u[0] = np.gradient(psi_filtered[2], axis=1) - np.gradient(psi_filtered[1], axis=2)  # u_x
        u[1] = np.gradient(psi_filtered[0], axis=2) - np.gradient(psi_filtered[2], axis=0)  # u_y
        u[2] = np.gradient(psi_filtered[1], axis=0) - np.gradient(psi_filtered[0], axis=1)  # u_z

        # --- 🎯 Normierung der Störung ---
        target_umax = 1
        umax = np.max(np.sqrt(np.sum(u ** 2, axis=0)))
        if umax > 0:
            u *= target_umax / umax

        # --- ➕ Basisströmung in x (Poiseuille-Profil) ---
        y_normalized = yg / yg.max()
        base_umax = 1.0
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

