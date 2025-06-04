"""
Boundary Conditions.

The `__call__` function of a boundary defines its application to the distribution functions.

Boundary conditions can define a mask (a boolean numpy array)
that specifies the grid points on which the boundary
condition operates.

Boundary classes can define two functions `make_no_stream_mask` and `make_no_collision_mask`
that prevent streaming and collisions on the boundary nodes.

The no-stream mask has the same dimensions as the distribution functions (Q, x, y, (z)) .
The no-collision mask has the same dimensions as the grid (x, y, (z)).

"""

import torch
import numpy as np
from lettuce import (LettuceException)

__all__ = ["BounceBackBoundary", "AntiBounceBackOutlet", "EquilibriumBoundaryPU", "EquilibriumOutletP"
    , "TGV3D", "newsuperTGV3D","superTGV2D", "HalfwayBounceBackBoundary", "WallFunctionBoundary"]


class BounceBackBoundary:
    """Fullway Bounce-Back Boundary"""

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice

    def __call__(self, f):
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)  # "Punkte an denen self.mask, also randpunkte liegen, werden mit f[] bezogen und andere mit f"
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


class EquilibriumBoundaryPU:
    """Sets distributions on this boundary to equilibrium with predefined velocity and pressure.
    Note that this behavior is generally not compatible with the Navier-Stokes equations.
    This boundary condition should only be used if no better options are available.
    """

    def __init__(self, mask, lattice, units, velocity, pressure=0):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.units = units
        self.velocity = lattice.convert_to_tensor(velocity)
        self.pressure = lattice.convert_to_tensor(pressure)

    def __call__(self, f):
        rho = self.units.convert_pressure_pu_to_density_lu(self.pressure)
        u = self.units.convert_velocity_to_lu(self.velocity)
        feq = self.lattice.equilibrium(rho, u)
        feq = self.lattice.einsum("q,q->q", [feq, torch.ones_like(f)])
        f = torch.where(self.mask, feq, f)
        return f


class AntiBounceBackOutlet:
    """Allows distributions to leave domain unobstructed through this boundary.
        Based on equations from page 195 of "The lattice Boltzmann method" (2016 by Krüger et al.)
        Give the side of the domain with the boundary as list [x, y, z] with only one entry nonzero
        [1, 0, 0] for positive x-direction in 3D; [1, 0] for the same in 2D
        [0, -1, 0] is negative y-direction in 3D; [0, -1] for the same in 2D
        """

    def __init__(self, lattice, direction):

        assert isinstance(direction, list), \
            LettuceException(
                f"Invalid direction parameter. Expected direction of type list but got {type(direction)}.")

        assert len(direction) in [1, 2, 3], \
            LettuceException(
                f"Invalid direction parameter. Expected direction of of length 1, 2 or 3 but got {len(direction)}.")

        assert (direction.count(0) == (len(direction) - 1)) and ((1 in direction) ^ (-1 in direction)), \
            LettuceException(
                "Invalid direction parameter. "
                f"Expected direction with all entries 0 except one 1 or -1 but got {direction}.")

        direction = np.array(direction)
        self.lattice = lattice

        # select velocities to be bounced (the ones pointing in "direction")
        self.velocities = np.concatenate(np.argwhere(np.matmul(self.lattice.stencil.e, direction) > 1 - 1e-6), axis=0)

        # build indices of u and f that determine the side of the domain
        self.index = []
        self.neighbor = []
        for i in direction:
            if i == 0:
                self.index.append(slice(None))
                self.neighbor.append(slice(None))
            if i == 1:
                self.index.append(-1)
                self.neighbor.append(-2)
            if i == -1:
                self.index.append(0)
                self.neighbor.append(1)
        # construct indices for einsum and get w in proper shape for the calculation in each dimension
        if len(direction) == 3:
            self.dims = 'dc, cxy -> dxy'
            self.w = self.lattice.w[self.velocities].view(1, -1).t().unsqueeze(1)
        if len(direction) == 2:
            self.dims = 'dc, cx -> dx'
            self.w = self.lattice.w[self.velocities].view(1, -1).t()
        if len(direction) == 1:
            self.dims = 'dc, c -> dc'
            self.w = self.lattice.w[self.velocities]

    def __call__(self, f):
        u = self.lattice.u(f)
        u_w = u[[slice(None)] + self.index] + 0.5 * (u[[slice(None)] + self.index] - u[[slice(None)] + self.neighbor])
        f[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = (
                - f[[self.velocities] + self.index] + self.w * self.lattice.rho(f)[[slice(None)] + self.index] *
                (2 + torch.einsum(self.dims, self.lattice.e[self.velocities], u_w) ** 2 / self.lattice.cs ** 4
                 - (torch.norm(u_w, dim=0) / self.lattice.cs) ** 2)
        )
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.array(self.lattice.stencil.opposite)[self.velocities]] + self.index] = 1
        return no_stream_mask

    # not 100% sure about this. But collisions seem to stabilize the boundary.
    # def make_no_collision_mask(self, f_shape):
    #    no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
    #    no_collision_mask[self.index] = 1
    #    return no_collision_mask


class EquilibriumOutletP(AntiBounceBackOutlet):
    """Equilibrium outlet with constant pressure.
    """

    def __init__(self, lattice, direction, rho0=1.0):
        super(EquilibriumOutletP, self).__init__(lattice, direction)
        self.rho0 = rho0

    def __call__(self, f):
        here = [slice(None)] + self.index
        other = [slice(None)] + self.neighbor
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        rho_w = self.rho0 * torch.ones_like(rho[here])
        u_w = u[other]
        f[here] = self.lattice.equilibrium(rho_w[..., None], u_w[..., None])[..., 0]
        return f

    def make_no_stream_mask(self, f_shape):
        no_stream_mask = torch.zeros(size=f_shape, dtype=torch.bool, device=self.lattice.device)
        no_stream_mask[[np.setdiff1d(np.arange(self.lattice.Q), self.velocities)] + self.index] = 1
        return no_stream_mask

    def make_no_collision_mask(self, f_shape):
        no_collision_mask = torch.zeros(size=f_shape[1:], dtype=torch.bool, device=self.lattice.device)
        no_collision_mask[self.index] = 1
        return no_collision_mask


class TGV3D:

    def __init__(self, lattice):
     self.lattice = lattice
    def __call__(self, f):



       for row in range(len(self.lattice.stencil.switch_xz)):
          self.saver=f[self.lattice.stencil.switch_xz[row][0],:,0,:].clone()
          f[self.lattice.stencil.switch_xz[row][0],:,0,:]=f[self.lattice.stencil.switch_xz[row][1],:,-1,:].clone()
          f[self.lattice.stencil.switch_xz[row][1],:,-1,:] = self.saver


       for row in range(len(self.lattice.stencil.switch_yz)):
          self.saver = f[self.lattice.stencil.switch_yz[row][0], 0, :, :].clone()
          f[self.lattice.stencil.switch_yz[row][0], 0, :, :] = f[self.lattice.stencil.switch_yz[row][1], -1, :, :].clone()
          f[self.lattice.stencil.switch_yz[row][1], -1, :, :] = self.saver




       for row in range(len(self.lattice.stencil.switch_x)):
           self.saver = torch.flip(f[self.lattice.stencil.switch_x[row][0],:,:,0].clone(),dims=[1])
           f[self.lattice.stencil.switch_x[row][0],:,:,0]=torch.flip(f[self.lattice.stencil.switch_x[row][1],:,:,-1].clone(),dims=[1])
           f[self.lattice.stencil.switch_x[row][1],:,:, -1]=self.saver


       return f


class newsuperTGV3D:
    def __init__(self, lattice):
        self.lattice = lattice

        self.e=self.lattice.stencil.e
        #####################################
        s_a = np.array([[0, -1, -1, 0, 1, 1, 1, 2],
                             [1, -1, 1, 0, -1, 1, 1, 2],
                             [2, 1, 1, 0, 1, 1, -1, 2],
                             [2, -1, 1, 1, 1, 0, -1, 2],
                             [0, 1, -1, 1, 1, 0, 1, 2],
                             [1, 1, 1, 1, -1, 0, 1, 2]])

        self.switch_stencil_wall = []

        for side in range(6):
            self.opposite = []
            for i in range(len(self.e)):
                for j in range(len(self.e)):
                    if self.e[i, s_a[side, 0]] == s_a[side, 1] and \
                            self.e[i, 0] == s_a[side, 2] * self.e[j, s_a[side, 3]] and \
                            self.e[i, 1] == s_a[side, 4] * self.e[j, s_a[side, 5]] and \
                            self.e[i, 2] == s_a[side, 6] * self.e[j, s_a[side, 7]]:
                        self.opposite.append((i, j))
            self.switch_stencil_wall.append(self.opposite)

        s_b = np.array([[0, -1, 1, 1, 0, -1, 1, 1, 2, 2],
                             [0, 1, 1, -1, 0, 1, 1, -1, 2, 2],
                             [0, 1, 1, 1, 0, -1, 1, -1, 2, 2],
                             [0, -1, 1, -1, 0, 1, 1, 1, 2, 2],
                             [1, -1, 2, 1, 1, 1, 2, -1, 0, 0],
                             [0, -1, 2, 1, 0, 1, 2, -1, 1, 1],
                             [1, 1, 2, 1, 0, -1, 2, -1, 0, 1],
                             [0, 1, 2, 1, 1, -1, 2, -1, 1, 0],
                             [1, 1, 2, -1, 1, -1, 2, 1, 0, 0],
                             [0, -1, 2, -1, 1, 1, 2, 1, 1, 0],
                             [1, -1, 2, -1, 0, 1, 2, 1, 0, 1],
                             [0, 1, 2, -1, 0, -1, 2, 1, 1, 1]])

        self.switch_stencil_borders = []

        for b in range(12):
            self.opposite = []
            for i in range(len(self.e)):
                for j in range(len(self.e)):
                    if self.e[i, s_b[b, 0]] == s_b[b, 1] and self.e[i, s_b[b, 2]] == s_b[b, 3] and \
                        self.e[j, s_b[b, 4]] == s_b[b, 5] and self.e[j, s_b[b, 6]] == s_b[b, 7] and \
                            self.e[i, s_b[b, 8]] == self.e[j, s_b[b, 9]]:
                        self.opposite.append((i, j))
            self.switch_stencil_borders.append(self.opposite)

        self.opposite = []
        self.switch_stencil_corner = []

        for i in range(len(self.e)):
            for j in range(len(self.e)):
                if self.e[i, 0] != 0 and self.e[i, 1] != 0 and self.e[i, 2] != 0 and self.e[i, 0] == -self.e[j, 0] and \
                        self.e[i, 1] == -self.e[j, 1] and self.e[i, 2] == -self.e[j, 2]:
                    self.opposite.append((i, j))
        self.switch_stencil_corner.append(self.opposite)

        self.swap_w= [[(0,slice(None),slice(None)),(-1,slice(None),slice(None))]
                                             ,[(slice(None),0,slice(None)),(slice(None),-1,slice(None))]
                                             ,[(slice(None),slice(None),-1),(slice(None),slice(None),0)]
                                             ,[(slice(None),slice(None),0),(slice(None),slice(None),-1)]
                                             ,[(-1,slice(None),slice(None)),(slice(None),0,slice(None))]
                                             ,[(slice(None),-1,slice(None)),(0,slice(None),slice(None))]]

        self.borders=[(0,-1,slice(None)),(-1,0,slice(None)),(0,0,slice(None)),(-1,-1,slice(None)),(slice(None),-1,0)
                                  ,(-1,slice(None),0),(0,slice(None),0),(slice(None),0,0)
            ,(slice(None),0,-1),(slice(None),-1,-1),
            (-1,slice(None),-1),(0,slice(None),-1)]
        self.corners=[(-1,-1,-1),(0,0,0),(-1,-1,0),(0,0,-1),(-1,0,-1),(0,-1,0),(-1,0,0),(0,-1,-1)]

    def __call__(self, f):

        self.f_copies=torch.stack((f[:,0,:,:].clone(),f[:,:,0,:].clone(),f[:,:,:,-1].clone(),
                                   f[:,:,:,0].clone(),f[:,-1,:,:].clone(),f[:,:,-1,:].clone()), dim=3)

        self.f_copies_borders=torch.stack((f[:,0,-1,:].clone(),f[:,-1,0,:].clone(),f[:,-1,-1,:].clone(),
                                            f[:,0,0,:].clone(),f[:,:,0,-1].clone()
                                            ,f[:,0,:,-1].clone(),f[:,:,-1,-1].clone(),
                                            f[:,-1,:,-1].clone(),f[:,:,-1,0].clone(),
                                            f[:,0,:,0].clone()
                                            ,f[:,:,0,0].clone(),f[:,-1,:,0].clone()),dim=2)

        self.f_copies_corners = torch.stack([f[:, -1, -1, -1].clone(), f[:,0,0,0].clone(),
                                             f[:,-1,-1,0].clone(), f[:,0,0,-1].clone(),
                                             f[:,-1,0,-1].clone(),f[:,0,-1,0].clone(),
                                             f[:, -1,0,0].clone(),f[:, 0,-1,-1].clone()],dim=1)

        for i in range(6):
            for j in range(len(self.switch_stencil_wall[i])):

                if i == 3:
                    f[self.switch_stencil_wall[i][j][1],*self.swap_w[i][1]] = \
                        torch.transpose(self.f_copies[self.switch_stencil_wall[i][j][0],:, :, i],0,1)
                else:

                    f[self.switch_stencil_wall[i][j][1],*self.swap_w[i][1]]=\
                        self.f_copies[self.switch_stencil_wall[i][j][0],:,:,i]

        for i in range(12):
            for j in range(len(self.switch_stencil_borders[i])):
                j=j
                f[self.switch_stencil_borders[i][j][1], *self.borders[i]] = \
                    self.f_copies_borders[self.switch_stencil_borders[i][j][0],:, i]

        if any(inner for inner in self.switch_stencil_corner):
            self.switch_stencil_corner = [(19, 20), (20, 19), (21, 22), (22, 21), (23, 24), (24, 26), (25, 23),
                                          (26, 25)]
            for i in range(8):
                index = self.switch_stencil_corner[i]

                i=i
                f[index[0],*self.corners[index[0]-19]]=\
                    self.f_copies_corners[index[1], index[1]-19]


        return f

class superTGV2D:

    def __init__(self, lattice):
     self.lattice = lattice
    def __call__(self, f):

        f_copie=f.clone()

        f[3,0,:]=f_copie[2,:,-1]
        f[6,0,:]=f_copie[5,:,-1]
        f[7,0,:]=f_copie[6,:,-1]

        f[7,:,0]=f_copie[8,-1,:]
        f[4,:,0]=f_copie[1,-1,:]
        f[8,:,0]=f_copie[5,-1,:]

        f[5,-1,:]=f_copie[6,0,:]
        f[1,-1,:]=f_copie[3,0,:]
        f[8,-1,:]=f_copie[7,0,:]

        f[5,:,-1]=f_copie[8,:,0]
        f[2,:,-1]=f_copie[4,:,0]
        f[6,:,-1]=f_copie[7,:,0]

        f[7,0,0]=f_copie[5,-1,-1]
        f[8,-1,0]=f_copie[8,-1,0]
        f[5,-1,-1]=f_copie[7,0,0]
        f[6,0,-1]=f_copie[6,0,-1]

        return f

class HalfwayBounceBackBoundary:

    def __init__(self, mask, lattice):
        self.mask = mask
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))

        self.f_index_fluid = []  # marks population from boundary-neighboring fluid node, pointing inside the boundary
        self.f_index_solid = []  # marks population from fluid-neighboring boundary node, pointing inside the boundary (stores f_collided for force calculation and bounce)

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self._no_stream_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
            self._no_stream_mask = self._no_stream_mask | self.mask
            a, b = np.where(mask)
            # np.arrays: list of (a) x-coordinates and (b) y-coordinates in the boundary.mask
            # ...to enable iteration over all boundary/wall/object-nodes
            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_index_fluid.append([self.lattice.stencil.opposite[i],
                                                 a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                 b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            self.f_index_solid.append([self.lattice.stencil.opposite[i], a[p], b[p]])

                            self._no_stream_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = False  # allows storage of bounce-relevant populations
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self._no_stream_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            self._no_stream_mask = self._no_stream_mask | self.mask
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            self.f_index_fluid.append([self.lattice.stencil.opposite[i],
                                                 a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                 b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                 c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            self.f_index_solid.append([self.lattice.stencil.opposite[i], a[p], b[p], c[p]])

                            self._no_stream_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = False  # allows storage of bounce-relevant populations
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_index_fluid = torch.tensor(np.array(self.f_index_fluid), device=self.lattice.device,
                                          dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_solid = torch.tensor(np.array(self.f_index_solid), device=self.lattice.device,
                                          dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        self.stencil_e_tensor_index = torch.tensor(self.lattice.e, device=self.lattice.device, dtype=torch.int64)


    def __call__(self, f):
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index_fluid[:, 0]],
              self.f_index_fluid[:, 1],
              self.f_index_fluid[:, 2]] = f[self.f_index_solid[:, 0],
                                            self.f_index_solid[:, 1],
                                            self.f_index_solid[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index_fluid[:, 0]],
              self.f_index_fluid[:, 1],
              self.f_index_fluid[:, 2],
              self.f_index_fluid[:, 3]] = f[self.f_index_solid[:, 0],
                                            self.f_index_solid[:, 1],
                                            self.f_index_solid[:, 2],
                                            self.f_index_solid[:, 3]]

        # calc force on boundary:
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self._no_stream_mask.shape == f_shape
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self._no_stream_mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f):
        if self.lattice.D == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_solid[:, 0],
                                                                 self.f_index_solid[:, 1],
                                                                 self.f_index_solid[:, 2]],
                                              self.lattice.e[self.f_index_solid[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index_solid[:, 0],
                                                                 self.f_index_solid[:, 1],
                                                                 self.f_index_solid[:, 2],
                                                                 self.f_index_solid[:, 3]],
                                              self.lattice.e[self.f_index_solid[:, 0]])

class WallFunctionBoundary:
    def __init__(self, mask, lattice, viscosity, y_lattice=1.0, kappa=0.41, B=5.2, switch_yplus=30, max_iter=20, tol=1e-6):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.viscosity = viscosity
        self.y_lattice = y_lattice
        self.kappa = kappa
        self.B = B
        self.switch_yplus = switch_yplus
        self.max_iter = max_iter
        self.tol = tol

    def spalding_law(self, y_plus):
        # numerisch invertiert: solve u_plus = y_plus + (1/kappa) * (exp(kappa*u_plus) - 1 - kappa*u_plus - 0.5*(kappa*u_plus)^2 - (1/6)*(kappa*u_plus)^3)
        u_plus = y_plus.clone()
        for _ in range(self.max_iter):
            ku = self.kappa * u_plus
            exp_ku = torch.exp(ku)
            f = u_plus - y_plus - (1/self.kappa) * (exp_ku - 1 - ku - 0.5*ku**2 - (1/6)*ku**3)
            df = 1 - (exp_ku - 1 - ku - 0.5*ku**2)  # Ableitung
            delta = f / df
            u_plus = u_plus - delta
            if torch.max(torch.abs(delta)) < self.tol:
                break
        return u_plus

    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)

        u_x = u[0]
        shifted_mask = torch.roll(self.mask, shifts=1, dims=1)
        du_dy = torch.zeros_like(u_x)
        du_dy[shifted_mask] = u_x[shifted_mask] / self.y_lattice

        u_tau = torch.sqrt(torch.abs(self.viscosity * du_dy))
        y_plus = (self.y_lattice * u_tau) / self.viscosity

        u_plus_log = self.spalding_law(y_plus)
        u_plus = torch.where(
            y_plus < self.switch_yplus,
            y_plus,
            u_plus_log
        )

        u_x_target = u_plus * u_tau

        D = self.lattice.D
        u_target = torch.zeros((D,) + u_x.shape, device=f.device, dtype=f.dtype)
        u_target[0] = torch.where(self.mask, u_x_target, torch.tensor(0.0, device=f.device, dtype=f.dtype))

        feq = self.lattice.equilibrium(rho, u_target)
        f = torch.where(self.mask, feq, f)
        return f

    def make_no_collision_mask(self, f_shape):
        return self.mask
