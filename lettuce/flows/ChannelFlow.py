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

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]

        # Grundströmung in x-Richtung (PU)
        u_char = np.array([self.units.characteristic_velocity_pu, 0.0])[..., None, None]
        u = (1 - self.mask.astype(float)) * u_char

        # 🌀 Sinusförmige Anfangsstörung in Querkomponente (y-Richtung)
        Lx = x[0].max()
        Ly = x[1].max()
        kx = 2 * np.pi / Lx

        # Störung mit 0.05 * Hauptgeschwindigkeit, moduliert über Höhe
        from scipy.fft import fft2, ifft2

        def generate_perturbation(shape, amplitude=0.1):
            # Weißes Rauschen → Spektrum manipulieren → Rücktransformieren
            noise = np.random.randn(*shape)
            kspace = fft2(noise)
            # Dämpfe hohe Frequenzen nicht komplett raus
            nx, ny = shape
            kx = np.fft.fftfreq(nx)
            ky = np.fft.fftfreq(ny)
            kkx, kky = np.meshgrid(kx, ky, indexing='ij')
            k = np.sqrt(kkx ** 2 + kky ** 2)
            spectrum = kspace * np.exp(-k * 5.0)  # spektraler Dämpfungsfaktor
            return amplitude * np.real(ifft2(spectrum))

        perturb = generate_perturbation(x[0].shape)
        u[1] += perturb * (1 - self.mask.astype(float))

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
        xg, yg, zg = grid  # [res_x, res_y, res_z]
        p = np.zeros_like(xg)[None, ...]  # Druckfeld

        u = np.zeros((3, *xg.shape), dtype=float)  # Geschwindigkeitsfeld

        # Gleichmäßige Strömung in x-Richtung
        u[0] = self.units.characteristic_velocity_pu * (1 - self.mask.astype(float))

        # Geometriegrößen
        Lx = xg.max() - xg.min()
        Ly = yg.max() - yg.min()
        Lz = zg.max() - zg.min()

        # ✴️ Bedingung: nahe unterer oder oberer Wand (z. B. y < 10% oder y > 90% von Ly)
        near_lower = yg < 0.1 * Ly
        near_upper = yg > 0.9 * Ly
        near_wall = (near_lower | near_upper)

        # Normierte Koordinaten
        x_norm = xg / Lx
        z_norm = zg / Lz

        # 🎯 Wandnahe Störung in u₂ (Querströmung)
        disturbance = 0.1 * np.sin(2 * np.pi * x_norm) * np.sin(2 * np.pi * z_norm)
        u[1] += disturbance * near_wall.astype(float)

        # Maske respektieren
        u *= (1 - self.mask.astype(float))

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

