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
        xg, yg, zg = grid  # [res_x, res_y, res_z]

        # Leicht variiertes Anfangsdruckfeld
        p = 1.0 + 0.001 * np.random.randn(*xg.shape)
        p = p[None, ...]  # Druckform [1, res_x, res_y, res_z]

        # Grundströmung in x-Richtung
        u = np.zeros((3, *xg.shape), dtype=float)
        u[0] = self.units.characteristic_velocity_pu * (1 - self.mask.astype(float))

        # Divergenzfreies Störfeld hinzufügen
        def generate_divergence_free_noise(shape, amplitude=0.05):
            from scipy.fft import fftn, ifftn
            nx, ny, nz = shape
            kx = np.fft.fftfreq(nx).reshape(-1, 1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1, 1)
            kz = np.fft.fftfreq(nz).reshape(1, 1, -1)
            k2 = kx ** 2 + ky ** 2 + kz ** 2
            k2[k2 == 0] = 1.0  # Singularität vermeiden

            vx = np.random.randn(nx, ny, nz)
            vy = np.random.randn(nx, ny, nz)
            vz = np.random.randn(nx, ny, nz)

            fx = fftn(vx)
            fy = fftn(vy)
            fz = fftn(vz)

            dot = kx * fx + ky * fy + kz * fz
            fx -= kx * dot / k2
            fy -= ky * dot / k2
            fz -= kz * dot / k2

            vx = np.real(ifftn(fx))
            vy = np.real(ifftn(fy))
            vz = np.real(ifftn(fz))

            return amplitude * np.stack([vx, vy, vz])

        np.random.seed(42)  # reproduzierbar
        perturb = generate_divergence_free_noise(xg.shape, amplitude=0.05)
        u += perturb * (1 - self.mask.astype(float))  # Maske respektieren

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

        # Gleichmäßige Strömung in x-Richtung (Poiseuille-ähnlich, falls gewünscht)
        u[0] = self.units.characteristic_velocity_pu * (1 - self.mask.astype(float))

        # Geometriegrößen
        Lx = xg.max() - xg.min()
        Ly = yg.max() - yg.min()
        Lz = zg.max() - zg.min()

        # Normierte Koordinaten
        x_norm = xg / Lx
        y_norm = yg / Ly
        z_norm = zg / Lz

        # 🎯 Wandgewichtung (glatt, symmetrisch bei y=0 und y=Ly)
        wall_weight = np.exp(-((y_norm - 0.05) ** 2) / 0.0025) + np.exp(-((y_norm - 0.95) ** 2) / 0.0025)

        # 🎯 Zufällige Modulation
        np.random.seed(42)
        rand_mod = 1.0 + 0.05 * np.random.randn(*xg.shape)

        # 🎯 Frequenzkombinationen
        fx = np.sin(2 * np.pi * x_norm) + np.sin(4 * np.pi * x_norm)
        fz = np.sin(2 * np.pi * z_norm) + np.sin(3 * np.pi * z_norm)

        # 🎯 Störungen hinzufügen
        amp = 0.1  # Amplitude der Störung

        u[0] += amp * fx * fz * wall_weight * rand_mod  # x-Richtung (stromabwärts)
        u[1] += amp * np.sin(2 * np.pi * z_norm) * wall_weight * rand_mod  # y-Richtung (quer)
        u[2] += amp * np.sin(6 * np.pi * x_norm) * wall_weight * rand_mod  # z-Richtung (spanwise)

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

