"""Lattice Boltzmann Solver"""

from timeit import default_timer as timer

import numpy

import lettuce
from lettuce import (
    LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, torch_gradient
)
from lettuce.util import pressure_poisson
import pickle
from copy import deepcopy
import warnings
import torch
import numpy as np

__all__ = ["Simulation", "SimulationHWBB"]

class Simulation:
    """High-level API for simulations.

    Attributes
    ----------
    reporters : list
        A list of reporters. Their call functions are invoked after every simulation step (and before the first one).

    """

    def __init__(self, flow, lattice, collision, streaming):
        self.flow = flow
        self.lattice = lattice
        self.collision = collision
        self.streaming = streaming
        self.i = 0

        grid = flow.grid
        p, u = flow.initial_solution(grid)
        assert list(p.shape) == [1] + list(grid[0].shape), \
            LettuceException(f"Wrong dimension of initial pressure field. "
                             f"Expected {[1] + list(grid[0].shape)}, "
                             f"but got {list(p.shape)}.")
        assert list(u.shape) == [lattice.D] + list(grid[0].shape), \
            LettuceException("Wrong dimension of initial velocity field."
                             f"Expected {[lattice.D] + list(grid[0].shape)}, "
                             f"but got {list(u.shape)}.")
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        self.reporters = []

        # Define masks, where the collision or streaming are not applied
        x = flow.grid
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool))
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))

        # Apply boundaries
        self._boundaries = self.flow.boundaries # store locally to keep the flow free from the boundary state !!ANGEPASSST
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
        if no_stream_mask.any():
            self.streaming.no_stream_mask = no_stream_mask

    def step(self, num_steps):

        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()

        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1

            self.f = self.streaming(self.f)

            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))

            for boundary in self._boundaries:
                self.f = boundary(self.f)
            self._report()



        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds



        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)

    def initialize(self, max_num_steps=500, tol_pressure=0.001):
        """Iterative initialization to get moments consistent with the initial velocity.

        Using the initialization does not better TGV convergence. Maybe use a better scheme?
        """
        warnings.warn("Iterative initialization does not work well and solutions may diverge. Use with care. "
                      "Use initialize_f_neq instead.",
                      ExperimentalWarning)
        transform = get_default_moment_transform(self.lattice)
        collision = BGKInitialization(self.lattice, self.flow, transform)
        streaming = self.streaming
        p_old = 0
        for i in range(max_num_steps):
            self.f = streaming(self.f)
            self.f = collision(self.f)
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.lattice.rho(self.f))
            if (torch.max(torch.abs(p - p_old))) < tol_pressure:
                break
            p_old = deepcopy(p)
        return i

    def initialize_pressure(self, max_num_steps=100000, tol_pressure=1e-6):
        """Reinitialize equilibrium distributions with pressure obtained by a Jacobi solver.
        Note that this method has to be called before initialize_f_neq.
        """
        u = self.lattice.u(self.f)
        rho = pressure_poisson(
            self.flow.units,
            self.lattice.u(self.f),
            self.lattice.rho(self.f),
            tol_abs=tol_pressure,
            max_num_steps=max_num_steps
        )
        self.f = self.lattice.equilibrium(rho, u)

    def initialize_f_neq(self, order=6):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
        rho = self.lattice.rho(self.f)
        u = self.lattice.u(self.f)

        grad_u0 = torch_gradient(u[0], dx=1, order=order)[None, ...]
        grad_u1 = torch_gradient(u[1], dx=1, order=order)[None, ...]
        S = torch.cat([grad_u0, grad_u1])

        if self.lattice.D == 3:
            grad_u2 = torch_gradient(u[2], dx=1, order=order)[None, ...]
            S = torch.cat([S, grad_u2])

        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2
        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)
        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        feq = self.lattice.equilibrium(rho, u)
        self.f = feq + fneq

    def analytical_gradient_tgv(self, x, y, z):
        """Berechnet den analytischen Gradienten des Taylor-Green Vortex Geschwindigkeitsfeldes."""
        grad = torch.zeros((3, 3, *x.shape), dtype=x.dtype, device=x.device)

        # ux Ableitungen
        grad[0, 0] = torch.cos(x) * torch.cos(y) * torch.cos(z)  # ∂ux/∂x
        grad[0, 1] = -torch.sin(x) * torch.sin(y) * torch.cos(z)  # ∂ux/∂y
        grad[0, 2] = -torch.sin(x) * torch.cos(y) * torch.sin(z)  # ∂ux/∂z

        # uy Ableitungen
        grad[1, 0] = torch.sin(x) * torch.sin(y) * torch.cos(z)  # ∂uy/∂x
        grad[1, 1] = -torch.cos(x) * torch.cos(y) * torch.cos(z)  # ∂uy/∂y
        grad[1, 2] = torch.cos(x) * torch.sin(y) * torch.sin(z)  # ∂uy/∂z

        # uz = 0, also sind die Ableitungen 0
        # grad[2, 0] = grad[2, 1] = grad[2, 2] = 0 (bereits durch torch.zeros gesetzt)

        return grad

    def initialize_f_neq_float32(self, order=6):
        """Initialize the distribution function values in Float32 precision, but keep computations in Float64."""

        # Float32-Genauigkeit erzwingen, aber in Float64 zurückwandeln
        def fake_float32(tensor):
            return tensor.round(decimals=7)
        self.f = fake_float32(self.f)
        rho = fake_float32(self.lattice.rho(self.f))
        u = fake_float32(self.lattice.u(self.f))

        grad_u0 = fake_float32(torch_gradient(u[0], dx=1, order=order)[None, ...])
        grad_u1 = fake_float32(torch_gradient(u[1], dx=1, order=order)[None, ...])
        S = torch.cat([grad_u0, grad_u1])

        if self.lattice.D == 3:
            grad_u2 = fake_float32(torch_gradient(u[2], dx=1, order=order)[None, ...])
            S = torch.cat([S, grad_u2])

        Pi_1 = fake_float32(1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2)

        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)

        Q = fake_float32(Q)  # Fake Float32 für Q
        Pi_1_Q = fake_float32(self.lattice.einsum('ab,iab->i', [Pi_1, Q]))
        fneq = fake_float32(self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q]))

        # Fake Float32 für rho, u und e
        rho = fake_float32(rho)
        u = fake_float32(u)
        e = fake_float32(self.lattice.e)

        feq = fake_float32(self.lattice.equilibrium(rho, u))  # Jetzt sind alle Inputs "fake" Float32
        self.f = fake_float32(feq + fneq)  # Float64 bleibt erhalten, da fake_float32 die Präzision nur abschneidet

    def initialize_fneq_with_analytical_gradient(self, mach):

        """Initialisiert f basierend auf dem analytischen Gradienten des TGV."""
        L = 2 * torch.pi
        N = self.f.shape[1]  # Annahme: f hat die Form [q, Nx, Ny, Nz]

        # Erstelle Gitterpunkte mit "endpoint=False"
        dx = L / N
        x = torch.arange(0, L, dx, device=self.f.device) + dx / 2
        y = torch.arange(0, L, dx, device=self.f.device) + dx / 2
        z = torch.arange(0, L, dx, device=self.f.device) + dx / 2

        # Erstelle das Gitter
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

        # Analytisches Geschwindigkeitsfeld
        rho = self.lattice.rho(self.f)  # Dichte ist typischerweise 1 im TGV

        u = self.lattice.u(self.f)

        # Berechne den analytischen Gradienten
        grad_u = (self.analytical_gradient_tgv(X, Y, Z)) * mach * 1 / torch.sqrt(
            torch.tensor(3.0, device=X.device)) * torch.pi * 2 / N

        # Berechnung von Pi_1 und fneq
        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * grad_u / self.lattice.cs ** 2
        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)
        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        # Berechnung von feq und f
        feq = self.lattice.equilibrium(rho, u)
        self.f = feq + fneq

    def save_checkpoint(self, filename):
        """Write f as np.array using pickle module."""
        with open(filename, "wb") as fp:
            pickle.dump(self.f, fp)

    def load_checkpoint(self, filename):
        """Load f as np.array using pickle module."""
        with open(filename, "rb") as fp:
            self.f = pickle.load(fp)


class SimulationHWBB(Simulation):
    def __init__(self, flow, lattice, collision, streaming):
        print("Calling SimulationHWBB.__init__")  # Debug-Ausgabe
        super().__init__(flow, lattice, collision, streaming)


    def step(self, num_steps):
        """Perform stream-and-collide steps with support for Halfway Bounce-Back."""
        start = timer()

        if self.i == 0:
            self._report()  # Call reporters before first step

        for _ in range(num_steps):
            ### 1. Collision
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))

            ### 2. Store f_collided for boundaries (needed by Halfway BB)
            for boundary in self._boundaries:
                if hasattr(boundary, "store_f_collided"):
                    boundary.store_f_collided(self.f)

            ### 3. Streaming
            self.f = self.streaming(self.f)

            ### 4. Apply Boundary Conditions (including Halfway BB)
            for boundary in self._boundaries:
                self.f = boundary(self.f)

            ### 5. Increase timestep counter
            self.i += 1

            ### 6. Reporting (e.g. for plots, monitoring)
            self._report()

        end = timer()

        ### Performance Calculation
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups
