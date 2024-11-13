import lettuce as lt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys
import warnings
from lettuce import (LettuceException)
import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, EquilibriumOutletP, AntiBounceBackOutlet
#from lettuce.flows.obstacleCylinder import ObstacleCylinder  # IMPORTED BELOW FOR STANDALONE SCRIPT
from collections import Counter
from lettuce.equilibrium import QuadraticEquilibrium_LessMemory

#from lettuce.max import draw_circular_mask  # IMPORTED BELOW FOR STANDALONE SCRIPT

import torch
import time
import datetime
import os
import psutil
import shutil
from pyevtk.hl import imageToVTK
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pickle
from copy import deepcopy
from timeit import default_timer as timer

warnings.simplefilter("ignore")


##################################################
##################################################
##################################################
# PUT IN YOUR PATHS TO storage, bulk-storage and refrerence-data dir:

output_path = "/home/bpicar3s/CylinderFlow"  # WHERE TO PUT DATA-FOLDER
scratch_dir = "/work/bpicar3s/CylinderFlow/Data"  # WHERE TP PUT VTK- and CPT-FILES (checkpoint, off by default)
diIlio_path = '/work/bpicar3s/CylinderFlow/Reference/'  # WHERE TO GET REFERENCE-DATA FOR AVG-VELOCITY-PROFILES FROM


##################################################
##################################################
##################################################
# CLASSES FOR STANDALONE USE:

class ObstacleCylinder:
    """
        add description here: unified version of 2D and 3D cylinder flow
        refined version of flow/obstacle.py, for cylinder-flow in 2D or 3D. The dimensions will be assumed from
        lattice.D

        Flow:
        - inflow (EquilibriumBoundaryPU) at x=0, outflow (EquilibriumOutletP) at x=xmax
        - further boundaries depend on parameters:
            - lateral (y direction): periodic, no-slip wall, slip wall
            - lateral (z direction): periodic (only if lattice.D==3)
        - obstacle: cylinder obstacle centered at (y_lu/2, y_LU/2), with radius, uniform symmetry in z-direction
            - obstacle mask has to be set externally?
        - boundary condition for obstacle can be chosen: hwbb, fwbb, ibb1
        - initial pertubation (trigger Von Kármán vortex street for Re>46) can be initialized in y and z direction
        - initial velocity can be 0, u_char or a parabolic profile (parabolic if lateral_walls = "bounceback")
        - inlet/inflow velocity can be uniform u_char or parabolic

        Parameters:
        ----------
        <to fill>
        ----------
    """
    def __init__(self, shape, reynolds_number, mach_number, lattice, char_length_pu, char_length_lu, char_velocity_pu=1,
                 lateral_walls='periodic', bc_type='fwbb', perturb_init=True, u_init=0,
                 x_offset=0, y_offset=0):
        # shape of the domain (2D or 3D):
        if len(shape) != lattice.D:
            raise ValueError(f"{lattice.D}-dimensional lattice requires {lattice.D}-dimensional `shape`")
        if len(shape) == 2:
            self.shape = (int(shape[0]), int(shape[1]))
        elif len(shape) == 3:
            self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        else:
            print("WARNING: shape is not 2- or 3-dimensional...(!)")
        #self.shape = shape

        self.char_length_pu = char_length_pu  # characteristic length

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu  # reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
        )

        # flow and boundary settings
        self.perturb_init = perturb_init  # toggle: introduce asymmetry in initial solution to trigger v'Karman Vortex Street
        self.u_init = u_init  # toggle: initial solution velocity profile type
        self.lateral_walls = lateral_walls  # toggle: lateral walls to be bounce back (bounceback), slip wall (slip) or periodic (periodic)
        self.bc_type = bc_type  # toggle: bounce back algorithm: halfway (hwbb) or fullway (fwbb)

        # initialize masks (init with zeros)
        self.solid_mask = np.zeros(shape=self.shape, dtype=bool)  # marks all solid nodes (obstacle, walls, ...)
        self.in_mask = np.zeros(self.grid[0].shape, dtype=bool)  # marks all inlet nodes
        self.wall_mask = np.zeros_like(self.solid_mask)  # marks lateral (top+bottom) walls
        self._obstacle_mask = np.zeros_like(self.solid_mask)  # marks all obstacle nodes (for fluid-solid-force_calc.)

        # cylinder geometry in LU (1-based indexing!)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.radius = char_length_lu / 2
        self.y_pos = self.shape[1] / 2 + 0.5 + self.y_offset  # y_position of cylinder-center in 1-based indexing
        self.x_pos = self.y_pos + self.x_offset  # keep symmetry of cylinder in x and y direction

        xyz = tuple(np.linspace(1, n, n) for n in self.shape)  # Tupel of index-lists (1-n (one-based!))
        if self.units.lattice.D == 2:
            x_lu, y_lu = np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y-index
        elif self.units.lattice.D == 3:
            x_lu, y_lu, z_lu = np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- and z-index
        else:
            print("WARNING: something went wrong in LU-gird-index generation, lattice.D must be 2 or 3!")

        condition = np.sqrt((x_lu - self.x_pos) ** 2 + (y_lu - self.y_pos) ** 2) < self.radius
        self.obstacle_mask[np.where(condition)] = 1
        self.solid_mask[np.where(condition)] = 1

        # indexing doesn't need z-Index for 3D, everything is broadcasted along z!
        if self.lateral_walls == 'bounceback' or self.lateral_walls == 'slip':  # if top and bottom are link-based BC
            self.wall_mask[:, [0, -1]] = True  # don't mark wall nodes as inlet
            self.solid_mask[np.where(self.wall_mask)] = 1  # mark solid walls
            self.in_mask[0, 1:-1] = True  # inlet on the left, except for top and bottom wall (y=0, y=y_max)
        else:  # if lateral_wals == 'periodic', no walls
            self.in_mask[0, :] = True  # inlet on the left (x=0)

        # generate parabolic velocity profile for inlet BC if lateral_walls (top and bottom) are bounce back walls (== channel-flow)
        self.u_inlet = self.units.characteristic_velocity_pu * self._unit_vector()  # u = [ux,uy,uz] = [1,0,0] in PU // uniform cahracteristic velocity in x-direction
        if self.lateral_walls == 'bounceback':
            ## parabolic velocity profile, zeroing on the edges
            ## How to parabola:
            ## 1.parabola in factoriezed form (GER: "Nullstellenform"): y = (x-x1)*(x-x2)
            ## 2.parabola with a maximum and zero at x1=0 und x2=x0: y=-x*(x-x0)
            ## 3.scale parabola, to make y_s(x_s)=1 the maximum: y=-x*(x-x0)*(1/(x0/2)²)
            ## (4. optional) scale amplitude with 1.5 to have a mean velocity of 1, also making the integral of a homogeneous velocity profile with u=1 and the parabolic profile being equal
            (nx, ny, nz) = self.shape  # number of gridpoints in y direction
            parabola_y = np.zeros((1, ny))
            y_coordinates = np.linspace(0, ny,
                                        ny)  # linspace() creates n points between 0 and ny, including 0 and ny:
            # top and bottom velocity values will be zero to agree with wall-boundary-condition
            parabola_y[:, 1:-1] = - 1.5 * np.array(self.u_inlet).max() * y_coordinates[1:-1] * (
                        y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2  # parabolic velocity profile
            # scale with 1.5 to achieve a mean velocity of u_char! -> DIFFERENT FROM cylinder2D and cylinder3D (!)
            if self.units.lattice.D == 2:
                # in 2D u1 needs Dimension 1 x ny (!)
                velocity_y = np.zeros_like(parabola_y)  # y-velocities = 0
                self.u_inlet = np.stack([parabola_y, velocity_y], axis=0)  # stack/pack u-field
            elif self.units.lattice.D == 3:
                ones_z = np.ones(nz)
                parabola_yz = parabola_y[:, :, np.newaxis] * ones_z
                parabola_yz_zeros = np.zeros_like(parabola_yz)
                # create u_xyz inlet yz-plane:
                self.u_inlet = np.stack([parabola_yz, parabola_yz_zeros, parabola_yz_zeros], axis=0)  # stack/pack u-field

    @property
    def obstacle_mask(self):
        return self._obstacle_mask

    @obstacle_mask.setter
    def obstacle_mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.shape
        self._obstacle_mask = m.astype(bool)
        # self.solid_mask[np.where(self._obstacle_mask)] = 1  # (!) this line is not doing what it should! solid_mask is now defined in the initial solution (see below)!

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        u_max_pu = self.units.characteristic_velocity_pu * self._unit_vector()
        u_max_pu = append_axes(u_max_pu, self.units.lattice.D)
        self.solid_mask[np.where(self.obstacle_mask)] = 1  # This line is needed, because the obstacle_mask.setter does not define the solid_mask properly (see above) #OLD
        ### initial velocity field: "u_init"-parameter
        # 0: uniform u=0
        # 1: uniform u=1 or parabolic (depends on lateral_walls -> bounceback => parabolic; slip, periodic => uniform)
        u = (1 - self.solid_mask) * u_max_pu
        if self.u_init == 0:
            u = u * 0  # uniform u=0
        else:
            if self.lateral_walls == 'bounceback':  # parabolic along y, uniform along x and z (similar to poiseuille-flow)
                ny = self.shape[1]  # number of gridpoints in y direction
                ux_factor = np.zeros(ny)  # vector for one column (u(x=0))
                # multiply parabolic profile with every column of the velocity field:
                y_coordinates = np.linspace(0, ny, ny)
                ux_factor[1:-1] = - y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2
                if self.units.lattice.D == 2:
                    u = np.einsum('k,ijk->ijk', ux_factor, u)
                elif self.units.lattice.D == 3:
                    u = np.einsum('k,ijkl->ijkl', ux_factor, u)
            else:  # lateral_walls == periodic or slip
                # initiale velocity u_PU=1 on every fluid node
                u = (1 - self.solid_mask) * u_max_pu

        ### perturb initial velocity field-symmetry (in y and z) to trigger 'von Karman' vortex street
        if self.perturb_init:  # perturb initial solution in y
            # overlays a sine-wave on the second column of nodes x_lu=1 (index 1)
            ny = x[1].shape[1]
            if u.max() < 0.5 * self.units.characteristic_velocity_pu:
                # add perturbation for small velocities
                #OLD 2D: u[0][1] += np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_pu * 1.0
                amplitude_y = np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.1
                if self.units.lattice.D == 2:
                    u[0][1] += amplitude_y
                elif self.units.lattice.D == 3:
                    nz = x[2].shape[2]
                    plane_yz = np.ones_like(u[0, 1])  # plane of ones
                    u[0][1] = np.einsum('y,yz->yz', amplitude_y, plane_yz)  # plane of amplitude in y
                    amplitude_z = np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.1  # amplitude in z
                   # print("amplitude y:", amplitude_y.shape)
                   # print("u[0][1]:", u[0][1].shape)
                   # print("amplitude z:", amplitude_z.shape)
                    # factor = 1 + np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * 0.3  # pertubation in z-direction
                    u[0][1] += np.einsum('z,yz->yz', amplitude_z, plane_yz)
            else:
                # multiply scaled down perturbation if velocity field is already near u_char
                #OLD 2D: u[0][1] *= 1 + np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * 0.3
                factor = 1 + np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * 0.1
                if self.units.lattice.D == 2:
                    u[0][1] *= factor
                elif self.units.lattice.D == 3:
                    nz = x[2].shape[1]
                    plane_yz = np.ones_like(u[0, 1, :, :])
                    u[0][1] = np.einsum('y,yz->yz', factor, u[0][1])
                    factor = 1 + np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * 0.1  # pertubation in z-direction
                    u[0][1] = np.einsum('z,yz->yz', factor, u[0][1])
        return p, u

    @property
    def grid(self):
        # THIS IS NOT USED AT THE MOMENT. QUESTION: SHOULD THIS BE ONE- OR ZERO-BASED? Indexing or "node-number"?
        xyz = tuple(self.units.convert_length_to_pu(np.linspace(0, n, n)) for n in self.shape)  # tuple of lists of x,y,(z)-values/indices
        return np.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)values/indices

    @property
    def boundaries(self):
        # inlet ("left side", x[0],y[1:-1], z[:])
        inlet_boundary = EquilibriumBoundaryPU(
            self.in_mask,
            self.units.lattice, self.units,
            # self.units.characteristic_velocity_pu * self._unit_vector())
            self.u_inlet)  # works with a 1 x D vector or an ny x D vector thanks to einsum-magic in EquilibriumBoundaryPU

        # lateral walls ("top and bottom walls", x[:], y[0,-1], z[:])
        lateral_boundary = None  # stays None if lateral_walls == 'periodic'
        if self.lateral_walls == 'bounceback':
            if self.bc_type == 'hwbb' or self.bc_type == 'HWBB':  # use halfway bounce back
                lateral_boundary = HalfwayBounceBackBoundary(self.wall_mask, self.units.lattice)
            else:  # else use fullway bounce back
                lateral_boundary = FullwayBounceBackBoundary(self.wall_mask, self.units.lattice)
        elif self.lateral_walls == 'slip' or self.bc_type == 'SLIP':  # use slip-walöl (symmetry boundary)
            lateral_boundary = SlipBoundary(self.wall_mask, self.units.lattice, 1)  # slip on x(z)-plane

        # outlet ("right side", x[-1],y[:], (z[:]))
        if self.units.lattice.D == 2:
            outlet_boundary = EquilibriumOutletP(self.units.lattice, [1, 0])  # outlet in positive x-direction
        else: # self.units.lattice.D == 3:
            outlet_boundary = EquilibriumOutletP(self.units.lattice, [1, 0, 0])  # outlet in positive x-direction

        # obstacle (for example: obstacle "cylinder" with radius centered at position x_pos, y_pos) -> to be set via obstacle_mask.setter
        obstacle_boundary = None
        # (!) the obstacle_boundary should alway be the last boundary in the list of boundaries to correctly calculate forces on the obstacle
        if self.bc_type == 'hwbb' or self.bc_type == 'HWBB':
            obstacle_boundary = HalfwayBounceBackBoundary(self.obstacle_mask, self.units.lattice)
        elif self.bc_type == 'ibb1' or self.bc_type == 'IBB1':
            obstacle_boundary = InterpolatedBounceBackBoundary(self.obstacle_mask, self.units.lattice,
                                                               x_center=(self.shape[1] / 2 - 0.5),
                                                               y_center=(self.shape[1] / 2 - 0.5), radius=self.radius)
        elif self.bc_type == 'ibb1c1':
            obstacle_boundary = InterpolatedBounceBackBoundary_compact_v1(self.obstacle_mask, self.units.lattice,
                                                               x_center=(self.shape[1] / 2 - 0.5),
                                                               y_center=(self.shape[1] / 2 - 0.5), radius=self.radius)
        elif self.bc_type == 'ibb1c2':
            obstacle_boundary = InterpolatedBounceBackBoundary_compact_v2(self.obstacle_mask, self.units.lattice,
                                                                          x_center=(self.shape[1] / 2 - 0.5),
                                                                          y_center=(self.shape[1] / 2 - 0.5),
                                                                          radius=self.radius)
        elif self.bc_type == 'fwbbc':
            obstacle_boundary = FullwayBounceBackBoundary_compact(self.obstacle_mask, self.units.lattice)
        elif self.bc_type == 'hwbbc1':
            obstacle_boundary = HalfwayBounceBackBoundary_compact_v1(self.obstacle_mask, self.units.lattice)
        elif self.bc_type == 'hwbbc2':
            obstacle_boundary = HalfwayBounceBackBoundary_compact_v2(self.obstacle_mask, self.units.lattice)
        elif self.bc_type == 'hwbbc3':
            obstacle_boundary = HalfwayBounceBackBoundary_compact_v3(self.obstacle_mask, self.units.lattice)
        else:  # use Fullway Bounce Back
            obstacle_boundary = FullwayBounceBackBoundary(self.obstacle_mask, self.units.lattice)

        if lateral_boundary is None:  # if lateral boundary is periodic...don't return a boundary-object
            return [
                inlet_boundary,
                outlet_boundary,
                obstacle_boundary
            ]
        else:
            return [
                inlet_boundary,
                outlet_boundary,
                lateral_boundary,
                obstacle_boundary
            ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]


class InterpolatedBounceBackBoundary:
    """Interpolated Bounce Back Boundary Condition first introduced by Bouzidi et al. (2001), as described in Kruger et al.
        (2017)
        - improvement of the simple bounce back (SBB) Algorithm, used in Fullway and/or Halfway Boucne Back (FWBB, HWBB)
        Boundary Conditions (see FullwayBounceBackBoundary and HalfwayBounceBackBoundary classes)
        - linear or quadratic interpolation of populations to retain the true boundary location between fluid- and
        solid-node

        * version 1.0: interpolation of a cylinder (circular in xy, axis along z). Axis position x_center, y_center with
        radius (ALL in LU!)
        NOTE: a mathematical condition for the boundary surface has to be known for calculation of the intersection point
        of boundary link and boundary surface for interpolation!
    """

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary) and considered for momentum exchange)
        if self.lattice.D == 2:
            nx, ny = self.mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
            # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary)
            #            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
            self.d = np.zeros_like(self.f_mask,
                                   dtype=float)  # d: [q,x,y] store the link-length per boundary-cutting link
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
                        if not self.mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                         b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = 1
                            # f_mask[q,x,y]

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p],i,d1,d2,px,py,cx,cy)
                            # distance from fluid node to the "true" boundary location
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = d1
                            elif d2 <= 1 and np.isreal(d2):
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny] = d2
                            else:
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p], b[p],
                                      self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = self.mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            #            self.force = np.zeros((nx, ny, nz, 3))
            self.d = np.zeros_like(self.f_mask,
                                   dtype=float)  # d: [q,x,y] store the link-length per boundary-cutting link
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
                        if not self.mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                         b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                         c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coodinate not needed for cylinder ! #pz = c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz  # fluid node z-coordinate

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder ! #cz = self.lattice.stencil.e[
                            #    self.lattice.stencil.opposite[i], 2]  # link-direction z to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)
                            # distance from fluid node to the "true" boundary location
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                       c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = d1
                            elif d2 <= 1 and np.isreal(d2):
                                self.d[self.lattice.stencil.opposite[i],
                                       a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                       b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                       c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = d2
                            else:
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p], b[p],
                                      c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.d = self.lattice.convert_to_tensor(self.d)
        self.f_collided = torch.zeros_like(self.f_mask, device=self.lattice.device, dtype=self.lattice.dtype)
        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):

        if self.interpolation_order == 2:
            print("warning: not implemented")
        else:  # interpolation_order==1:
            # f_tmp = f_collided[i,x_b]_interpolation before bounce
            f_tmp = torch.where(self.d <= 0.5,  # if d<=1/2
                                2 * self.d * self.f_collided + (1 - 2 * self.d) * f,
                                # interpolate from second fluid node
                                (1 / (2 * self.d)) * self.f_collided + (1 - 1 / (2 * self.d)) * self.f_collided[
                                    self.lattice.stencil.opposite])  # else: interpolate from opposing populations on x_b
            # (?) 1-1/(2d) ODER (2d-1)/2d, welches ist numerisch exakter?
            # f_collided an x_f entspricht f_streamed an x_b, weil entlang des links ohne collision gestreamt wird!
            # ... d.h. f_collided[i,x_f] entspricht f[i,x_b]
            f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_tmp[self.lattice.stencil.opposite], f)
            # HWBB: f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        # momentum exchange according to Bouzidi et al. (2001), equation 11.8 in Kruger et al. (2017) p.445 // watch out for wrong signs. Kruger states "-", but "+" gives correct result
        # ...original paper (Bouzidi 2001) and Ginzburg followup (2003) state a "+" as well...
        # tmp = torch.where(self.f_mask, f_collided, torch.zeros_like(f_bounced)) \
        #      - torch.where(self.f_mask, f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced))
        # tmp = torch.where(self.f_mask, f_collided - f_bounced[self.lattice.stencil.opposite], torch.zeros_like(f_bounced)) #WRONG
        tmp = torch.where(self.f_mask, self.f_collided + f_bounced[self.lattice.stencil.opposite],
                          torch.zeros_like(f_bounced))  # RIGHT
        self.force_sum = torch.einsum('i..., id -> d', tmp,
                                      self.lattice.e)  # CALCULATE FORCE / v3.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)

    def store_f_collided(self, f_collided):
        self.f_collided = torch.clone(f_collided)


class InterpolatedBounceBackBoundary_compact_v1:

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index_lt = []  # indices of relevant populations (for bounce back and force-calculation) with d<=0.5
        self.f_index_gt = []  # indices of relevant populations (for bounce back and force-calculation) with d>0.5
        self.d_lt = []  # distances between node and boundary for d<0.5
        self.d_gt = []  # distances between node and boundary for d>0.5

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny),
                                   dtype=bool)  # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary), needed to collect f_collided in simulation
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[
                                            1] * ny] = 1  # f_mask[q,x,y], marks all fs which point from fluid to solid (boundary)

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coodinate not needed for cylinder !

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder !

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,z,ci", a[p],
                                      b[p], c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.f_index_lt = torch.tensor(np.array(self.f_index_lt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(np.array(self.f_index_gt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.d_lt = self.lattice.convert_to_tensor(np.array(self.d_lt))
        self.d_gt = self.lattice.convert_to_tensor(np.array(self.d_gt))
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

        self.f_collided = []

        if lattice.D == 2:
            fc_q, fc_x, fc_y = torch.where(
                self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y))
        if lattice.D == 3:
            fc_q, fc_x, fc_y, fc_z = torch.where(
                self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y, fc_z))

        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):

        # BOUNCE (Bouzidi et al. (2001), as described in Kruger et al. (2017))
        if self.lattice.D == 2:
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                                  self.f_index_lt[:, 1],
                                                                                  self.f_index_lt[:, 2]] \
                                       + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                 self.f_index_lt[:, 1],
                                                                 self.f_index_lt[:, 2]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                          self.f_index_gt[:, 1],
                                                                                          self.f_index_gt[:, 2]] \
                                       + (1 - 1 / (2 * self.d_gt)) * self.f_collided.to_dense()[
                                           self.opposite_tensor[self.f_index_gt[:, 0]],
                                           self.f_index_gt[:, 1],
                                           self.f_index_gt[:, 2]]
        if self.lattice.D == 3:
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2],
              self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                                  self.f_index_lt[:, 1],
                                                                                  self.f_index_lt[:, 2],
                                                                                  self.f_index_lt[:, 3]] \
                                       + (1 - 2 * self.d_lt) * f[self.f_index_lt[:, 0],
                                                                 self.f_index_lt[:, 1],
                                                                 self.f_index_lt[:, 2],
                                                                 self.f_index_lt[:, 3]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2],
              self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                          self.f_index_gt[:, 1],
                                                                                          self.f_index_gt[:, 2],
                                                                                          self.f_index_gt[:, 3]] \
                                       + (1 - 1 / (2 * self.d_gt)) * self.f_collided.to_dense()[
                                           self.opposite_tensor[self.f_index_gt[:, 0]],
                                           self.f_index_gt[:, 1],
                                           self.f_index_gt[:, 2],
                                           self.f_index_gt[:, 3]]

        # CALCULATE FORCE
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opposite])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                                      self.f_index_lt[:, 1],
                                                                                      self.f_index_lt[:, 2]] \
                                          + f_bounced[self.opposite_tensor[self.f_index_lt[:, 0]],
                                                      self.f_index_lt[:, 1],
                                                      self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                        self.f_index_gt[:, 1],
                                                                                        self.f_index_gt[:, 2]] \
                                            + f_bounced[self.opposite_tensor[self.f_index_gt[:, 0]],
                                                        self.f_index_gt[:, 1],
                                                        self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_lt[:, 0],
                                                                                      self.f_index_lt[:, 1],
                                                                                      self.f_index_lt[:, 2],
                                                                                      self.f_index_lt[:, 3]] \
                                          + f_bounced[self.opposite_tensor[self.f_index_lt[:, 0]],
                                                      self.f_index_lt[:, 1],
                                                      self.f_index_lt[:, 2],
                                                      self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index_gt[:, 0],
                                                                                        self.f_index_gt[:, 1],
                                                                                        self.f_index_gt[:, 2],
                                                                                        self.f_index_gt[:, 3]] \
                                            + f_bounced[self.opposite_tensor[self.f_index_gt[:, 0]],
                                                        self.f_index_gt[:, 1],
                                                        self.f_index_gt[:, 2],
                                                        self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2]],
                                                                  size=f_collided.size()))
        if self.lattice.D == 3:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2], self.fc_index[3]],
                                                                  size=f_collided.size()))


class InterpolatedBounceBackBoundary_compact_v2:

    def __init__(self, mask, lattice, x_center, y_center, radius, interpolation_order=1):
        t_init_start = time.time()
        self.interpolation_order = interpolation_order
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index_lt = []  # indices of relevant populations (for bounce back and force-calculation) with d<=0.5
        self.f_index_gt = []  # indices of relevant populations (for bounce back and force-calculation) with d>0.5
        self.d_lt = []  # distances between node and boundary for d<0.5
        self.d_gt = []  # distances between node and boundary for d>0.5

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,ci", a[p],
                                      b[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:

                            # calculate intersection point of boundary surface and link ->
                            # ...calculate distance between fluid node and boundary surface on the link
                            px = a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx  # fluid node x-coordinate
                            py = b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny  # fluid node y-coordinate
                            # Z-coodinate not needed for cylinder !

                            cx = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 0]  # link-direction x to solid node
                            cy = self.lattice.stencil.e[
                                self.lattice.stencil.opposite[i], 1]  # link-direction y to solid node
                            # Z-coodinate not needed for cylinder !

                            # pq-formula
                            h1 = (px * cx + py * cy - cx * x_center - cy * y_center) / (cx * cx + cy * cy)  # p/2
                            h2 = (px * px + py * py + x_center * x_center + y_center * y_center
                                  - 2 * px * x_center - 2 * py * y_center - radius * radius) / (
                                         cx * cx + cy * cy)  # q

                            d1 = - h1 + np.sqrt(h1 * h1 - h2)
                            d2 = - h1 - np.sqrt(h1 * h1 - h2)

                            # print("xb,yb,i,d1,d2 xf, yf, cx, cy:", a[p], b[p], i, d1, d2, px, py, cx, cy)

                            # distance from fluid node to the "true" boundary location
                            # choose correct d and assign d and f_index
                            if d1 <= 1 and np.isreal(d1):  # d should be between 0 and 1

                                if d1 <= 0.5:
                                    self.d_lt.append(d1)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d1)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])

                            elif d2 <= 1 and np.isreal(d2):  # d should be between 0 and 1

                                if d2 <= 0.5:
                                    self.d_lt.append(d2)
                                    self.f_index_lt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                                else:  # d>0.5
                                    self.d_gt.append(d2)
                                    self.f_index_gt.append([self.lattice.stencil.opposite[i],
                                                            a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                            b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                            c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                            else:  # neither d1 or d2 is real and between 0 and 1
                                print("IBB WARNING: d1 is", d1, "; d2 is", d2, "for boundaryPoint x,y,z,ci", a[p],
                                      b[p], c[p], self.lattice.stencil.e[i, 0], self.lattice.stencil.e[i, 1],
                                      self.lattice.stencil.e[i, 2])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index_lt = torch.tensor(np.array(self.f_index_lt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.f_index_gt = torch.tensor(np.array(self.f_index_gt), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.d_lt = self.lattice.convert_to_tensor(np.array(self.d_lt))
        self.d_gt = self.lattice.convert_to_tensor(np.array(self.d_gt))
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

        f_collided_lt = torch.zeros_like(self.d_lt)  # float-tensor with number of (x_b nodes with d<=0.5) values
        f_collided_gt = torch.zeros_like(self.d_gt)  # float-tensor with number of (x_b nodes with d>0.5) values
        f_collided_lt_opposite = torch.zeros_like(self.d_lt)
        f_collided_gt_opposite = torch.zeros_like(self.d_gt)
        self.f_collided_lt = torch.stack((f_collided_lt, f_collided_lt_opposite), dim=1)
        self.f_collided_gt = torch.stack((f_collided_gt, f_collided_gt_opposite), dim=1)

        print("IBB initialization took " + str(time.time() - t_init_start) + "seconds")

    def __call__(self, f):
        ## f_collided_lt = [f_collided_lt, f_collided_lt.opposite] (!) in compact storage-layout

        if self.lattice.D == 2:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[
                self.f_index_lt[:, 0],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        if self.lattice.D == 3:
            # BOUNCE
            # if d <= 0.5
            f[self.opposite_tensor[self.f_index_lt[:, 0]],
              self.f_index_lt[:, 1],
              self.f_index_lt[:, 2],
              self.f_index_lt[:, 3]] = 2 * self.d_lt * self.f_collided_lt[:, 0] + (1 - 2 * self.d_lt) * f[
                self.f_index_lt[:, 0],
                self.f_index_lt[:, 1],
                self.f_index_lt[:, 2],
                self.f_index_lt[:, 3]]
            # if d > 0.5
            f[self.opposite_tensor[self.f_index_gt[:, 0]],
              self.f_index_gt[:, 1],
              self.f_index_gt[:, 2],
              self.f_index_gt[:, 3]] = (1 / (2 * self.d_gt)) * self.f_collided_gt[:, 0] + (
                        1 - 1 / (2 * self.d_gt)) * self.f_collided_gt[:, 1]

        # CALC. FORCE on boundary (MEM, MEA)
        self.calc_force_on_boundary(f)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f_bounced):
        ### force = e * (f_collided + f_bounced[opp.])
        if self.lattice.D == 2:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2]],
                                            self.lattice.e[self.f_index_gt[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = torch.einsum('i..., id -> d',
                                          self.f_collided_lt[:, 0] + f_bounced[
                                              self.opposite_tensor[self.f_index_lt[:, 0]],
                                              self.f_index_lt[:, 1],
                                              self.f_index_lt[:, 2],
                                              self.f_index_lt[:, 3]],
                                          self.lattice.e[self.f_index_lt[:, 0]]) \
                             + torch.einsum('i..., id -> d',
                                            self.f_collided_gt[:, 0] + f_bounced[
                                                self.opposite_tensor[self.f_index_gt[:, 0]],
                                                self.f_index_gt[:, 1],
                                                self.f_index_gt[:, 2],
                                                self.f_index_gt[:, 3]],
                                            self.lattice.e[self.f_index_gt[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2]])  # y
            self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:, 0]],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2]])  # y

            self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2]])  # y
            self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:, 0]],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2]])  # y
        if self.lattice.D == 3:
            self.f_collided_lt[:, 0] = torch.clone(f_collided[self.f_index_lt[:, 0],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2],  # y
                                                              self.f_index_lt[:, 3]])  # z
            self.f_collided_lt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_lt[:, 0]],  # q
                                                              self.f_index_lt[:, 1],  # x
                                                              self.f_index_lt[:, 2],  # y
                                                              self.f_index_lt[:, 3]])  # z

            self.f_collided_gt[:, 0] = torch.clone(f_collided[self.f_index_gt[:, 0],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2],  # y
                                                              self.f_index_gt[:, 3]])  # z
            self.f_collided_gt[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index_gt[:, 0]],  # q
                                                              self.f_index_gt[:, 1],  # x
                                                              self.f_index_gt[:, 2],  # y
                                                              self.f_index_gt[:, 3]])  # z


class FullwayBounceBackBoundary:
    """Fullway Bounce-Back Boundary (with added force_on_boundary calculation)
    - fullway = inverts populations within two substeps
    - call() must be called after Streaming substep
    - calculates the force on the boundary:
        - calculation is done after streaming, but theoretically the force is evaluated based on the populations touching/crossing the boundary IN this streaming step
    """

    # based on Master-Branch "class BounceBackBoundary"
    # added option to calculate force on the boundary by Momentum Exchange Method

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)  # which nodes are solid
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
#            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
            a, b = np.where(mask)
                # np.arrays: list of (a) x-indizes and (b) y-indizes in the boundary.mask
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the solid p
                            #OLD: self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]] = 1
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
#            self.force = np.zeros((nx, ny, nz, 3))
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
                        # if not mask[a[p] + self.lattice.stencil.e[i, 0]*single_layer[p, 0] - border_direction[p, 0]*nx,
                        #            b[p] + self.lattice.stencil.e[i, 1]*single_layer[p, 1] - border_direction[p, 1]*ny,
                        #            c[p] + self.lattice.stencil.e[i, 2]*single_layer[p, 2] - border_direction[p, 2]*nz]:
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:
                            # OLD: self.f_mask[self.lattice.stencil.opposite[i], a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]] = 1
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
#                if (b[p] == int(ny / 2)):
#                    print("node ({},{}): m={}, bdz={}, slz={}".format(str(a[p]), str(c[p]), mask[a[p], b[p], c[p]], border_direction[p, 2], single_layer[p, 2]))
#            print("border_direction\n", border_direction)
#            print("single_layer\n", single_layer)
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)

    def __call__(self, f):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:
        self.calc_force_on_boundary(f)
        # bounce (invert populations on boundary nodes)
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask

    def calc_force_on_boundary(self, f):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, f, torch.zeros_like(f))  # all populations f in the fluid region, which point at the boundary
        #self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # v1.1 - M.Kliemank
        #self.force = dx ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / dx  # v.1.2 - M.Bille (dt=dx, dx as a parameter)
        self.force_sum = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v2.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)
            # explanation for 2D:
                # sums forces in x and in y (and z) direction,
                # tmp: all f, that are marked in f_mask
                    # tmp.size: 9 x nx x ny (for 2D)
                # self.lattice.e: 9 x 2 (for 2D)
                # - the multiplication of f_i and c_i is down through the first dimension (q) = direction, indexname i
                # - the sign is given by the coordinates of the stencil-vectors (e[0 to 8] for 2D)
                # -> results in two dimensional output (index d) for x- and y-direction (for 2D)
                # "dx**self-lattice.D" = dx³ (3D) or dx² (2D) as prefactor, converting momentum density to momentum
                    # theoretically DELTA P (difference in momentum density) is calculated
                    # assuming smooth momentum transfer over dt, force can be calculated through: F= dP/dt
                    # ...that's why theoretically dividing by dt=dx=1 is necessary (BUT: c_i=1=dx/dt=1 so that can be omitted (v2.0) !)

        # # calculate Force on all boundary nodes individually:
        # if self.lattice.D == 2:
        #     self.force = 2 * torch.einsum('qxy, qd -> xyd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, direction (0=x, 1=y)]
        # if self.lattice.D == 3:
        #     self.force = 2 * torch.einsum('qxyz, qd -> xyzd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, z-coodrinate, direction (0=x, 1=y, 2=z)]


class FullwayBounceBackBoundary_compact:

    def __init__(self, mask, lattice):
        self.mask = mask  # which nodes are solid
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)

        self.f_index = []
        self.bounce_index = []

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y

                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
            a, b = np.where(mask)
                # np.arrays: list of (a) x-indizes and (b) y-indizes in the boundary.mask
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the solid p

                            self.f_index.append([self.lattice.stencil.opposite[i], a[p], b[p]])  # list of [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape

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
                            self.f_index.append([self.lattice.stencil.opposite[i], a[p], b[p], c[p]])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device, dtype=torch.int64) # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

    def __call__(self, f):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:
        self.calc_force_on_boundary(f)
        # bounce (invert populations on boundary nodes)
        # f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = f[self.f_index[:, 0],
                                      self.f_index[:, 1],
                                      self.f_index[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = f[self.f_index[:, 0],
                                      self.f_index[:, 1],
                                      self.f_index[:, 2],
                                      self.f_index[:, 3]]
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self, f):
        if self.lattice.D == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index[:, 0],
                                                                 self.f_index[:, 1],
                                                                 self.f_index[:, 2]], self.lattice.e[self.f_index[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', f[self.f_index[:, 0],
                                                                 self.f_index[:, 1],
                                                                 self.f_index[:, 2],
                                                                 self.f_index[:, 3]], self.lattice.e[self.f_index[:, 0]])


class HalfwayBounceBackBoundary:
    """Halfway Bounce Back Boundary (with added force_on_boundary calculation)
    - halfway = inverts populations within one substep
    - call() must be called after Streaming substep
    - calculates the force on the boundary:
        - calculation is done after streaming, but theoretically the force is evaluated based on the populations touching/crossing the boundary IN this streaming step
    """

    def __init__(self, mask, lattice):
        self.mask = mask
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
                # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary)
#            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny] = 1
                            # f_mask[q,x,y]
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
#            self.force = np.zeros((nx, ny, nz, 3))
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
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.f_collided = torch.zeros_like(self.f_mask, device=self.lattice.device, dtype=self.lattice.dtype)

    def __call__(self, f):
        # HALFWAY-BB: overwrite all populations (on fluid nodes) which came from boundary with pre-streaming populations (on fluid nodes) which pointed at boundary
            #print("f_mask:\n", self.f_mask)
            #print("f_mask(q2,x1,y1):\n", self.f_mask[2, 1, 1])
            #print("f_mask(q2,x1,y3):\n", self.f_mask[2, 1, 3])
            #print("f_mask(opposite):\n", self.f_mask[self.lattice.stencil.opposite])
        # calc force on boundary:
        self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        f = torch.where(self.f_mask[self.lattice.stencil.opposite], self.f_collided[self.lattice.stencil.opposite], f)
            # ersetze alle "von der boundary kommenden" Populationen durch ihre post-collision_pre-streaming entgegengesetzten Populationen
            # ...bounce-t die post_collision/pre-streaming Populationen an der Boundary innerhalb eines Zeitschrittes
            # ...von außen betrachtet wird "während des streamings", innerhalb des gleichen Zeitschritts invertiert.
            # (?) es wird keine no_streaming_mask benötigt, da sowieso alles, was aus der boundary geströmt käme hier durch pre-Streaming Populationen überschrieben wird.
            # ...ist das so, oder entsteht dadurch "Strömung" innerhalb des Obstacles? Diese hat zwar keinen direkten Einfluss auf die Größen im Fluidbereich,
            # ... lässt aber in der Visualisierung Werte ungleich Null innerhalb von Objekten entstehen und Mittelwerte etc. könnten davon beeinflusst werden. (?)
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
            # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
            # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post-processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, self.f_collided, torch.zeros_like(self.f_collided))  # all populations f in the fluid region, which point at the boundary
        #self.force = 1 ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / 1.0  # v1.1 - M.Kliemank
        #self.force = dx ** self.lattice.D * 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e) / dx  # v.1.2 - M.Bille (dt=dx, dx as a parameter)
        self.force_sum = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v2.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)
            # explanation for 2D:
                # sums forces in x and in y (and z) direction,
                # tmp: all f, that are marked in f_mask
                    # tmp.size: 9 x nx x ny (for 2D)
                # self.lattice.e: 9 x 2 (for 2D)
                # - the multiplication of f_i and c_i is down through the first dimension (q) = direction, indexname i
                # - the sign is given by the coordinates of the stencil-vectors (e[0 to 8] for 2D)
                # -> results in two dimensional output (index d) for x- and y-direction (for 2D)
                # "dx**self-lattice.D" = dx³ (3D) or dx² (2D) as prefactor, converting momentum density to momentum
                    # theoretically DELTA P (difference in momentum density) is calculated
                    # assuming smooth momentum transfer over dt, force can be calculated through: F= dP/dt
                    # ...that's why theoretically dividing by dt=dx=1 is necessary (BUT: c_i=1=dx/dt=1 so that can be omitted (v2.0) !)

        # # calculate Force on all boundary nodes individually:
        # if self.lattice.D == 2:
        #     self.force = 2 * torch.einsum('qxy, qd -> xyd', tmp,
        #                                   self.lattice.e)  # force = [x-coordinate, y-coodrinate, direction (0=x, 1=y)]
        # if self.lattice.D == 3:
        #     self.force = 2 * torch.einsum('qxyz, qd -> xyzd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, z-coodrinate, direction (0=x, 1=y, 2=z)]

    def store_f_collided(self, f_collided):
        self.f_collided = torch.clone(f_collided)


class HalfwayBounceBackBoundary_compact_v1:

    def __init__(self, mask, lattice):
        self.mask = mask
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which point from fluid to solid (boundary))

        self.f_index = []

        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny),
                                   dtype=bool)  # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary), needed to collect f_collided in simulation
                # f_mask: [q, nx, ny], marks all fs which point from fluid to solid (boundary)

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
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[
                                            1] * ny] = 1  # f_mask[q,x,y], marks all fs which point from fluid to solid (boundary)

                            self.f_index.append([self.lattice.stencil.opposite[i],
                                                a[p] + self.lattice.stencil.e[i, 0] - border[0]*nx,
                                                b[p] + self.lattice.stencil.e[i, 1] - border[1]*ny])

                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
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
                            self.f_mask[self.lattice.stencil.opposite[i],
                                        a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                        b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                        c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz] = 1
                            self.f_index.append([self.lattice.stencil.opposite[i],
                                                a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)
        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device, dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor

        self.f_collided = []

        if lattice.D == 2:
            fc_q, fc_x, fc_y = torch.where(self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y))
        if lattice.D == 3:
            fc_q, fc_x, fc_y, fc_z = torch.where(self.f_mask + self.f_mask[self.lattice.stencil.opposite])  # which populations of f_collided to store
            self.fc_index = torch.stack((fc_q, fc_x, fc_y, fc_z))

    def __call__(self, f):
        # calc force on boundary:
        self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = self.f_collided.to_dense()[self.f_index[:, 0],
                                                          self.f_index[:, 1],
                                                          self.f_index[:, 2]]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = self.f_collided.to_dense()[self.f_index[:, 0],
                                                          self.f_index[:, 1],
                                                          self.f_index[:, 2],
                                                          self.f_index[:, 3]]
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
            # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
            # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post-processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        if self.lattice.D == 2:
            self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index[:, 0],
                                                                                     self.f_index[:, 1],
                                                                                     self.f_index[:, 2]],
                                              self.lattice.e[self.f_index[:, 0]])
        if self.lattice.D == 3:
            self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided.to_dense()[self.f_index[:, 0],
                                                                                     self.f_index[:, 1],
                                                                                     self.f_index[:, 2],
                                                                                     self.f_index[:, 3]],
                                              self.lattice.e[self.f_index[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2]],
                                                                  size=f_collided.size()))
        if self.lattice.D == 3:
            self.f_collided = torch.clone(torch.sparse_coo_tensor(indices=self.fc_index,
                                                                  values=f_collided[self.fc_index[0], self.fc_index[1],
                                                                                    self.fc_index[2], self.fc_index[3]],
                                                                  size=f_collided.size()))


class HalfwayBounceBackBoundary_compact_v2:

    def __init__(self, mask, lattice):
        self.mask = mask  # location of solid-nodes
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(
            self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))

        self.f_index = []

        # searching boundary-fluid-interface and append indices to f_index, distance to boundary to d
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            a, b = np.where(mask)  # x- and y-index of boundaryTRUE nodes for iteration over boundary area

            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    # check for boundary-nodes neighboring the domain-border.
                    # ...they have to take the periodicity into account...
                    border = np.zeros(self.lattice.D, dtype=int)

                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left [x]
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right [x]
                        border[0] = 1

                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left [y]
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right [y]
                        border[1] = 1

                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the neighbour

                            self.f_index.append([self.lattice.stencil.opposite[i],
                                                 a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                 b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            a, b, c = np.where(mask)

            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    border = np.zeros(self.lattice.D, dtype=int)
                    # x - direction
                    if a[p] == 0 and self.lattice.stencil.e[i, 0] == -1:  # searching border on left
                        border[0] = -1
                    elif a[p] == nx - 1 and self.lattice.e[i, 0] == 1:  # searching border on right
                        border[0] = 1
                    # y - direction
                    if b[p] == 0 and self.lattice.stencil.e[i, 1] == -1:  # searching border on left
                        border[1] = -1
                    elif b[p] == ny - 1 and self.lattice.e[i, 1] == 1:  # searching border on right
                        border[1] = 1
                    # z - direction
                    if c[p] == 0 and self.lattice.stencil.e[i, 2] == -1:  # searching border on left
                        border[2] = -1
                    elif c[p] == nz - 1 and self.lattice.e[i, 2] == 1:  # searching border on right
                        border[2] = 1

                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                    b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                    c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz]:


                                self.f_index.append([self.lattice.stencil.opposite[i],
                                                     a[p] + self.lattice.stencil.e[i, 0] - border[0] * nx,
                                                     b[p] + self.lattice.stencil.e[i, 1] - border[1] * ny,
                                                     c[p] + self.lattice.stencil.e[i, 2] - border[2] * nz])
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there

        # convert relevant tensors:
        self.f_index = torch.tensor(np.array(self.f_index), device=self.lattice.device,
                                       dtype=torch.int64)  # the batch-index has to be integer
        self.opposite_tensor = torch.tensor(self.lattice.stencil.opposite, device=self.lattice.device,
                                            dtype=torch.int64)  # batch-index has to be a tensor
        f_collided = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        f_collided_opposite = torch.zeros_like(self.f_index[:, 0], dtype=self.lattice.dtype)
        self.f_collided = torch.stack((f_collided, f_collided_opposite), dim=1)

    def __call__(self, f):
        # calc force on boundary:
        self.calc_force_on_boundary()
        # bounce (invert populations on fluid nodes neighboring solid nodes)
        # f = torch.where(self.f_mask[self.lattice.stencil.opposite], f_collided[self.lattice.stencil.opposite], f)

        if self.lattice.D == 2:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2]] = self.f_collided[:, 0]
        if self.lattice.D == 3:
            f[self.opposite_tensor[self.f_index[:, 0]],
              self.f_index[:, 1],
              self.f_index[:, 2],
              self.f_index[:, 3]] = self.f_collided[:, 0]
        return f

    def make_no_stream_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]  # all dimensions of f except the 0th (q)
        # no_stream_mask has to be dimensions: (q,x,y,z) (z optional), but CAN be (x,y,z) (z optional).
        # ...in the latter case, torch.where broadcasts the mask to (q,x,y,z), so ALL q populations of a lattice-node are marked equally
        return self.lattice.convert_to_tensor(self.mask)

    def make_no_collision_mask(self, f_shape):
        # INFO: for the halfway bounce back boundary, a no_collision_mask ist not necessary, because the no_streaming_mask
        # ...prevents interaction between nodes inside and outside of the boundary region.
        # INFO: pay attention to the initialization of observable/moment-fields (u, rho,...) on the boundary nodes,
        # ...in the initial solution of your flow, especially if visualization or post processing uses the field-values
        # ...in the whole domain (including the boundary region)!
        assert self.mask.shape == f_shape[1:]
        return self.lattice.convert_to_tensor(self.mask)

    def calc_force_on_boundary(self):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
        # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
        # ...populations pointing at the surface of the boundary
        self.force_sum = 2 * torch.einsum('i..., id -> d', self.f_collided[:, 0], self.lattice.e[self.f_index[:, 0]])

    def store_f_collided(self, f_collided):
        if self.lattice.D == 2:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2]])  # y
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:,0]],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2]])  # y
        if self.lattice.D == 3:
            self.f_collided[:, 0] = torch.clone(f_collided[self.f_index[:, 0],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2],  # y
                                                          self.f_index[:, 3]])  # z
            self.f_collided[:, 1] = torch.clone(f_collided[self.opposite_tensor[self.f_index[:,0]],  # q
                                                          self.f_index[:, 1],  # x
                                                          self.f_index[:, 2],  # y
                                                          self.f_index[:, 3]])  # z


class HalfwayBounceBackBoundary_compact_v3:

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


class SlipBoundary:
    """bounces back in a direction given as 0, 1, or 2 for x, y, or z, respectively
        based on fullway bounce back algorithm (population remains in the wall for 1 time step)
    """

    def __init__(self, mask, lattice, direction):
        self.mask = lattice.convert_to_tensor(mask)
        self.lattice = lattice
        self.bb_direction = direction
        e = self.lattice.stencil.e
        bb_direction = self.bb_direction
        opposite_stencil = np.array(e)
        opposite_stencil[:, bb_direction] = -e[:, bb_direction]
        self.opposite = []
        for opp_dir in opposite_stencil:
            self.opposite.append(np.where(np.array(e == opp_dir).all(axis=1))[0][0])

    def __call__(self, f):
        f = torch.where(self.mask, f[self.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask


def draw_circular_mask(lattice, gridpoints_per_diameter, output_data=False, filebase=".", print_data=False):
    ### calculate and export 2D obstacle_mask as .png
    grid_x = gridpoints_per_diameter + 2
    if output_data:
        output_file = open(filebase + "/obstacle_mask_info.txt", "a")
        output_file.write("GPD = " + str(gridpoints_per_diameter) + "\n")
    if print_data:
        print("GPD = " + str(gridpoints_per_diameter))
    # define radius and position for a symetrical circular Cylinder-Obstacle
    radius_LU = 0.5 * gridpoints_per_diameter
    y_pos_LU = 0.5 * grid_x + 0.5
    x_pos_LU = y_pos_LU

    # get x,y,z meshgrid of the domain (LU)
    xyz = tuple(np.linspace(1, n, n) for n in (grid_x, grid_x))  # Tupel aus Listen indizes (1-n (nicht 0-based!))
    xLU, yLU = np.meshgrid(*xyz, indexing='ij')  # meshgrid aus den x-, y- (und z-)Indizes -> * damit man die einzelnen Einträge des Tupels übergibt, und nicht das eine Tupel

    # define cylinder (LU) (circle)
    obstacle_mast_for_visualization = np.sqrt((xLU - x_pos_LU) ** 2 + (yLU - y_pos_LU) ** 2) < radius_LU

    nx, ny = obstacle_mast_for_visualization.shape  # Anzahl x-Punkte, Anzahl y-Punkte (Skalar), (der gesamten Domain)

    rand_mask = np.zeros((nx, ny), dtype=bool)  # für Randpunkte, die es gibt
    rand_mask_f = np.zeros((lattice.Q, nx, ny), dtype=bool)  # für Randpunkte (inkl. Q-Dimension)
    rand_xq = []  # Liste aller x Werte (inkl. q-multiplizität)
    rand_yq = []  # Liste aller y Werte (inkl. q-multiplizität)

    a, b = np.where(obstacle_mast_for_visualization)  # np.array: Liste der (a) x-Koordinaten  und (b) y-Koordinaten der obstacle_mast_for_visualization
    # ...um über alle Boundary/Objekt/Wand-Knoten iterieren zu können
    for p in range(0, len(a)):  # für alle TRUE-Punkte der obstacle_mast_for_visualization
        for i in range(0, lattice.Q):  # für alle stencil-Richtungen c_i (hier lattice.stencil.e)
            try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                if not obstacle_mast_for_visualization[a[p] + lattice.stencil.e[i, 0], b[p] + lattice.stencil.e[i, 1]]:
                    # falls in einer Richtung Punkt+(e_x, e_y; e ist c_i) False ist, ist das also ein Oberflächenpunkt des Objekts (selbst True mit Nachbar False)
                    rand_mask[a[p], b[p]] = 1
                    rand_mask_f[lattice.stencil.opposite[i], a[p], b[p]] = 1
                    rand_xq.append(a[p])
                    rand_yq.append(b[p])
            except IndexError:
                pass  # just ignore this iteration since there is no neighbor there
    rand_x, rand_y = np.where(rand_mask)  # Liste aller Rand-x- und y-Koordinaten
    x_pos = sum(rand_x) / len(rand_x)  # x_Koordinate des Kreis-Zentrums
    y_pos = sum(rand_y) / len(rand_y)  # y-Koordinate des Kreis-Zentrums

    # calculate all radii and r_max and r_min
    r_max = 0
    r_min = gridpoints_per_diameter
    radii = np.zeros_like(rand_x, dtype=float)  # Liste aller Radien (ohne q) in LU
    for p in range(0, len(rand_x)):  # für alle Punkte
        radii[p] = np.sqrt(
            (rand_x[p] - x_pos) ** 2 + (rand_y[p] - y_pos) ** 2)  # berechne Abstand des Punktes zum Zentrum
        if radii[p] > r_max:
            r_max = radii[p]
        if radii[p] < r_min:
            r_min = radii[p]

    # calculate all radii (with q-multiplicity)
    radii_q = np.zeros_like(rand_xq, dtype=float)
    for p in range(0, len(rand_xq)):
        radii_q[p] = np.sqrt((rand_xq[p] - x_pos) ** 2 + (rand_yq[p] - y_pos) ** 2)

    ### all relative radii in relation to gpd/2
    radii_relative = radii / (
                radius_LU - 0.5)  # (substract 0.5 because "true" boundary location is 0.5LU further out than node-coordinates)
    radii_q_relative = radii_q / (radius_LU - 0.5)

    # calc. mean rel_radius
    r_rel_mean = sum(radii_relative) / len(radii_relative)
    rq_rel_mean = sum(radii_q_relative) / len(radii_q_relative)

    ## AREA calculation
    area_theory = np.pi * (gridpoints_per_diameter / 2) ** 2  # area = pi*r² in LU²
    area = len(a)  # area in LU = number of nodes, because every node has a cell of 1LU x 1LU around it

    if output_data:
        output_file.write("\nr_rel_mean: " + str(sum(radii_relative) / len(radii_relative)))
        output_file.write("\nrq_rel_mean: " + str(sum(radii_q_relative) / len(radii_q_relative)))
        output_file.write("\nr_rel_min: " + str(r_max / (radius_LU - 0.5)))
        output_file.write("\nr_rel_max: " + str(r_min / (radius_LU - 0.5)))
        output_file.write("\n\narea_rel: " + str(area / area_theory))

        output_file.write("\n\nradii: " + str(Counter(radii)))
        output_file.write("\nradii_q: " + str(Counter(radii_q)) + "\n\n")
        output_file.close()
    if print_data:
        print("area_rel: " + str(area / area_theory))

    ### PLOT Mask
    plt.figure()
    plt.imshow(obstacle_mast_for_visualization)
    # plt.xticks(np.arange(gridpoints_per_diameter + 2), minor=True)
    # plt.yticks(np.arange(gridpoints_per_diameter + 2), minor=True)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymax, ymin = ax.get_ylim()
    if gridpoints_per_diameter >= 10:
        plt.xticks(np.arange(0, xmax, int(xmax / 10)))
        plt.yticks(np.arange(0, ymax, int(ymax / 10)))
    else:
        plt.xticks(np.arange(0, xmax, 1))
        plt.yticks(np.arange(0, ymax, 1))
    plt.title("GPD = " + str(gridpoints_per_diameter))
    ax.set_xticks(np.arange(-.5, xmax, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ymax, 1), minor=True)

    # grid thickness, cicrle, node marker
    x, y = np.meshgrid(np.linspace(0, int(xmax), int(xmax + 1)), np.linspace(0, int(ymax), int(ymax + 1)))
    if gridpoints_per_diameter < 30:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=2)
        circle = plt.Circle((xmax / 2 - 0.25, ymax / 2 - 0.25), gridpoints_per_diameter / 2, color='r', fill=False,
                            linewidth=1)
        ax.add_patch(circle)
        plt.plot(x, y, marker='.', linestyle='', color="b", markersize=1)
    elif gridpoints_per_diameter < 70:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=1)
        circle = plt.Circle((xmax / 2 - 0.25, ymax / 2 - 0.25), gridpoints_per_diameter / 2, color='r', fill=False,
                            linewidth=0.5)
        ax.add_patch(circle)
    elif gridpoints_per_diameter < 100:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=0.5)
    elif gridpoints_per_diameter < 150:
        ax.grid(which="minor", color="k", axis='both', linestyle='-', linewidth=0.25)

    if output_data:
        plt.savefig(filebase + "/obstacle_mask_GPD" + str(gridpoints_per_diameter) + ".png")
    if print_data:
        plt.show()
    else:
        plt.close()


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError


class DragCoefficient(Observable):
    """The drag coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
    modified version of M.Kliemank's Drag Coefficient Code

    calculates the density, gets the force in x direction on the obstacle boundary,
    calculates the coefficient of drag
    """

    def __init__(self, lattice, flow, obstacle_boundary, area):
        super().__init__(lattice, flow)
        self.obstacle_boundary = obstacle_boundary
        self.area_lu = area * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.lattice.D-1) # crosssectional area of obstacle in LU (! lengthdimension in 2D -> area-dimension = self.lattice.D-1)
        self.nan = self.lattice.convert_to_tensor(torch.nan)
        self.solid_mask = self.lattice.convert_to_tensor(self.flow.solid_mask)

    def __call__(self, f):
        #rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding boundary region):
        rho_tmp = torch.where(self.solid_mask, self.nan, self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
        force_x_lu = self.obstacle_boundary.force_sum[0]  # get current force on obstacle in x direction
        drag_coefficient = force_x_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # calculate drag_coefficient in LU
        return drag_coefficient


class LiftCoefficient(Observable):
    """The lift coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
        modified version of M.Kliemank's lift Coefficient Code

        calculates the density, gets the force in y direction on the obstacle boundary,
        calculates the coefficient of lift
        """

    def __init__(self, lattice, flow, obstacle_boundary, area):
        super().__init__(lattice, flow)
        self.obstacle_boundary = obstacle_boundary
        self.area_lu = area * (self.flow.units.characteristic_length_lu / self.flow.units.characteristic_length_pu) ** (
                    self.lattice.D - 1)
        self.nan = self.lattice.convert_to_tensor(torch.nan)
        self.solid_mask = self.lattice.convert_to_tensor(self.flow.solid_mask)

    def __call__(self, f):
        #rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding boundary region):
        rho_tmp = torch.where(self.solid_mask, self.nan, self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
        force_y_lu = self.obstacle_boundary.force_sum[1] # get current force on obstacle in y direction
        lift_coefficient = force_y_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # calculate lift_coefficient in LU
        return lift_coefficient


class NaNReporter:
    """reports any NaN and aborts the simulation"""
    def __init__(self, flow, lattice, n_target, t_target, interval=100, ):
        self.flow = flow
        self.lattice = lattice
        self.interval = interval
        self.n_target = n_target
        self.t_target = t_target

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            if torch.isnan(f).any() == True:
                if self.lattice.D == 2:
                    q, x, y = torch.where(torch.isnan(f))
                    q = self.lattice.convert_to_numpy(q)
                    x = self.lattice.convert_to_numpy(x)
                    y = self.lattice.convert_to_numpy(y)
                    nan_location = np.stack((q, x, y), axis=-1)
                    print("(!) NaN detected at (q,x,y):", nan_location)
                if self.lattice.D == 3:
                    q, x, y, z = torch.where(torch.isnan(f))
                    q = self.lattice.convert_to_numpy(q)
                    x = self.lattice.convert_to_numpy(x)
                    y = self.lattice.convert_to_numpy(y)
                    z = self.lattice.convert_to_numpy(z)
                    nan_location = np.stack((q, x, y, z), axis=-1)
                    print("(!) NaN detected at (q,x,y,z):", nan_location)
                print("(!) NaN detected in time step", i, "of", self.n_target, "(interval:", self.interval, ")")
                print("(!) Aborting simulation at t_PU", self.flow.units.convert_time_to_pu(i), "of", self.t_target)
                sys.exit()


class AverageVelocityReporter:
    """Reports the streamwise velocity averaged in span direction (z) at defined position in x.
        Refer to Di Ilio et al. 2018 for further references
        out = averaged u profile at position
        t_out = t_LU and t_PU
        """
    def __init__(self, lattice, flow, position, interval=1, start=1):
        self.lattice = lattice
        self.flow = flow
        self.interval = interval
        self.start = start  # step to start with reporting
        self.t_out = []
        self.out = []
        self.x_position = int(round(position, 0))  # rounded LU position

        # linear interpolation of u for x_pos off grid
        if position % 1 != 0:
            self.interpol = True
            self.x_pos1 = int(np.floor(position))
            self.x_pos2 = int(np.ceil(position))
            self.w1 = position - self.x_pos1
            self.w2 = 1 - self.w1

    def __call__(self, i, t, f):
        if i % self.interval == 0 and i >= self.start:
            if self.interpol:
                u = self.lattice.u(f)[:, self.x_pos1] * self.w1 + self.lattice.u(f)[:, self.x_pos2] * self.w2
            else:
                u = self.lattice.u(f)[:, self.x_position]
            u = self.flow.units.convert_velocity_to_pu(u).cpu().numpy()
            self.t_out.append([i, t])
            if self.lattice.D == 2:
                self.out.append(u)
            elif self.lattice.D == 3:
                self.out.append(np.mean(u, axis=2))


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
        self.i = 0  # index of the current timestep

        # M.Bille:
        self.store_f_collided = []  # toggle if f is stored after collision and not overwritten through streaming,
        # ...f_collided might be needed together with f_collided_and_streamed for boundary-conditions or calculation of
        # ...momentum exchange (force on boundary, coefficient of drag etc.)
        self.times = [[], [], [], [], []]  # list of lists for time-measurement (collision, streaming, boundary, reporters)
        self.time_avg = dict()
        self.t_max = 72*3600-10*60  # max. runtime 71:50:00 h

        # CALCULATE INITIAL SOLUTION of flow and CHECK initial solution for correct dimensions
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

        # INITIALIZE distribution function f: convert u and rho from numpy to torch.tensor
        u = lattice.convert_to_tensor(flow.units.convert_velocity_to_lu(u))
        rho = lattice.convert_to_tensor(flow.units.convert_pressure_pu_to_density_lu(p))
        self.f = lattice.equilibrium(rho, lattice.convert_to_tensor(u))

        # list of reporters
        self.reporters = []

        # Define masks, where collision or streaming are not applied
        # (initialized with 0, later specified by e.g. boundary conditions)
        x = flow.grid  # meshgrid, dimensions: D x nx x ny (x nz)
        self.no_collision_mask = lattice.convert_to_tensor(np.zeros_like(x[0], dtype=bool))  # dimensions: nx x ny (x nz)
        no_stream_mask = lattice.convert_to_tensor(np.zeros(self.f.shape, dtype=bool))
            # "self" and "no self" because no_stream_mask is written to streaming-object in the init,
            # ... no_collision_mask is used in the simulation.step()

        # retrieve no-streaming and no-collision markings from all boundaries
        self._boundaries = deepcopy(self.flow.boundaries)  # store locally to keep the flow free from the boundary state -> WHY?
        for boundary in self._boundaries:
            if hasattr(boundary, "make_no_collision_mask"):
                # get no-collision markings from boundaries
                self.no_collision_mask = self.no_collision_mask | boundary.make_no_collision_mask(self.f.shape)
            if hasattr(boundary, "make_no_stream_mask"):
                # get no-streaming markings from boundaries
                no_stream_mask = no_stream_mask | boundary.make_no_stream_mask(self.f.shape)
        if no_stream_mask.any():
            # write no_streaming_mask to streaming-object
            self.streaming.no_stream_mask = no_stream_mask

        # define f_collided (post-collision, pre-streaming f) storage format for hwbb, ibb1, ibb1c1 or ibb1c2, ...
        self.boundary_range = range(len(self._boundaries))
        for boundary_index in self.boundary_range:
            if hasattr(self._boundaries[boundary_index], "store_f_collided"):
                self.store_f_collided.append(True)
                self._boundaries[boundary_index].store_f_collided(self.f)
            else:
                self.store_f_collided.append(False)

    def step(self, num_steps):
        """ Take num_steps stream-and-collision steps and return performance in MLUPS.
        M.Bille: added force_calculation on object/boundaries
        M.Bille: added halfway bounce back boundary
        """
        start = timer()
        if self.i == 0:  # if this is the first timestep, calc. initial force on Object/walls/boundary/obstacle and call reporters
            # reporters are called before the first timestep
            self._report()
        for _ in range(num_steps):  # simulate num_step timesteps
            time1 = timer()

            ### COLLISION
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            # ...and store post-collision population for halfway-bounce-back boundary condition
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))

            time2 = timer()

            ### STORE f_collided FOR BOUNDARIES needing pre-streaming populations for bounce or force-calculation
            for boundary_index in self.boundary_range:
                if self.store_f_collided[boundary_index]:
                    self._boundaries[boundary_index].store_f_collided(self.f)

            time3 = timer()

            ### STREAMING
            self.f = self.streaming(self.f)

            time4 = timer()

            ### BOUNDARY
            # apply boundary conditions
            for boundary in self._boundaries:
                self.f = boundary(self.f)

            # count step
            self.i += 1

            time5 = timer()
            # call reporters
            self._report()

            time6 = timer()
            self.times[0].append(time2-time1)  # time to collide
            self.times[1].append(time3-time2)  # time to store f_collided
            self.times[2].append(time4-time3)  # time to stream
            self.times[3].append(time5-time4)  # time to boundary
            self.times[4].append(time6-time5)  # time to report

            if time6-start > self.t_max:  # if T_total > 71:50:00 h
                num_steps = _
                break
                # TODO: print out real number of steps! sim.i - i_start
        end = timer()

        # calculate individual runtimes (M.Bille)
        if num_steps > 0:
            self.time_avg = dict(time_collision=sum(self.times[0])/len(self.times[0]),
                                 time_store_f_collided=sum(self.times[1])/len(self.times[1]),
                                 time_streaming=sum(self.times[2])/len(self.times[2]),
                                 time_boundary=sum(self.times[3])/len(self.times[3]),
                                 time_reporter=sum(self.times[4])/len(self.times[4]))
        else:  # no division by zero
            self.time_avg = dict(time_collision=-1,
                                 time_store_f_collided=-1,
                                 time_streaming=-1,
                                 time_boundary=-1,
                                 time_reporter=-1)

        # calculate runtime and performance in MLUPS
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

    def _report(self):
        for reporter in self.reporters:
            reporter(self.i, self.flow.units.convert_time_to_pu(self.i), self.f)

    def save_checkpoint(self, filename, device=None):
        """Write f as np.array using pickle module."""
        if device is not None:
            # to store checkpoint through cpu with "device='cpu'",
            # ...because on multi-gpu systems the checkpoint is loaded into the device-index it was stored on,
            # ...making multi-gpu multi-simulation runs tricky, because the device has to be ajusted after loading the
            # ...checkpoint. Recommend: copy f to cpu before checkpointing,
            # ...and loading back into gpu on loading checkpoint (see load_checkpoint() below)
            f_store = self.f.to(device, copy=True)
            with open(filename, "wb") as fp:
                pickle.dump(f_store, fp)
        else:  # if no device is given, checkpoint retains/contains device-affinity
            with open(filename, "wb") as fp:
                pickle.dump(self.f, fp)

    def load_checkpoint(self, filename, device=None):
        """Load f as np.array using pickle module."""
        if device is not None:
            with open(filename, "rb") as fp:
                f_load = pickle.load(fp)
                self.f = f_load.to(device, copy=False)
        else:  # if no device is given, device from checkponit is used. May run into issues if the device of the ckeckpoint is different from the device of the rest of the simulation.
            with open(filename, "rb") as fp:
                self.f = pickle.load(fp)


##################################################
##################################################
##################################################
# START OF SCRIPT from HERE #

##################################################
# ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0")
parser.add_argument("--gpd", default=20, type=int, help="number of gridpoints per diameter")
parser.add_argument("--dpx", default=0, type=int, help="domain length in diameters")
parser.add_argument("--dpy", default=19, type=int, help="domain height in diameters")
parser.add_argument("--dpz", default=0, type=float, help="domain width in diameters")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate")
parser.add_argument("--collision", default="bgk", help="collision operator (bgk, kbc, reg)")
parser.add_argument("--lateral_walls", default='periodic', help="boundary condition in y direction (periodic, bounceback, slip)")
parser.add_argument("--bc_type", default='fwbb', help="bounce back algorithm (fwbb, hwbb, ibb1)")
parser.add_argument("--name", default="3Dcylinder", help="name of the simulation, appears in output directory name")
parser.add_argument("--stencil", default="D3Q27", help="stencil (D3Q27, D3Q19, D3Q15)")
parser.add_argument("--output_vtk", default=False, help="output vtk-data with 10 fps (large!)")
parser.add_argument("--device", default="cuda:0", help="cuda-device for multi-gpu nodes cuda:[0,1,2,3]")
parser.add_argument("--calcUProfiles", default=False,help="calculate average velocity profiles as in Di Ilio et al. 2018 and output plots and time-averages data for plots")
parser.add_argument("--outputUProfiles", default=False, help="output average velocity profiles over time (full timeseries)")
parser.add_argument("--nan_reporter", default=False, help="stop simulation if NaN is detected in f field")
parser.add_argument("--EQLM", default=False, help="use equilibrium_lessMemory, saved up to 20% GPU-VRAM but takes 2% longer runtime on GPU")
parser.add_argument("--from_cpt", default=False, help="start from checkpoint. (!) provide --cpt_file path")
parser.add_argument("--cpt_file", default=None, help="path and name of cpt_file to use if --from_cpt=True")
parser.add_argument("--sim_i", default=0, type=int, help="step index of last checkpoints-step to start from for time-indexing of observables etc.")
parser.add_argument("--write_cpt", default=False, help="write checkpoint after finishing simulation")

args = vars(parser.parse_args())

##################################################
# PARAMETERS

re = args["re"]  # Reynoldszahl
Ma = 0.05  # Machzahl
n_steps = args["n_steps"]  # Schrittzahl
setup_diameter = 1  # D_PU = char_length_pu -> this defines the PU-Reference
flow_velocity = 1  # U_PU = char_velocity_pu -> this defines the PU-Reference velocity (u_max or u_mean of inflow)

if re > 1000:
    periodic_start = 0.4
else:
    periodic_start = 0.9  # relative starting point of peak_finding for Cd_mean Measurement to cut of any transients

gridpoints_per_diameter = args["gpd"]  # gp_per_D -> this defines the resolution ( D_LU = GPD+1)
domain_height_in_D = args["dpy"]  # D/Y  -> this defines the domain-size and total number of Lattice-Nodes
if args["dpx"] == 0:
    domain_length_in_D = 2 * domain_height_in_D  # D/X = domain length in X- / flow-direction
else:
    domain_length_in_D = args["dpx"]
domain_width_in_D = args[
    "dpz"]  # D/Z = DpZ = diameters per domain width in Z-direction -> domain size in periodic 3rd dimension
if args["dpz"] == 0:  # no z-deptth -> create pseudo 2D-domain with 3D stencil
    domain_width_in_D = 1 / gridpoints_per_diameter

# if DpY is even, resulting GPD can't be odd for symmetrical cylinder and channel
# ...if DpY is even, GPD will be corrected to be even for symmetrical cylinder
# ...use odd DpY to use odd GPD
gpd_correction = False
if domain_height_in_D % 2 == 0 and gridpoints_per_diameter % 2 != 0:
    gpd_correction = True  # gpd_was_corrected-flag
    gpd_setup = gridpoints_per_diameter  # store old gpd for output
    gridpoints_per_diameter = int(gridpoints_per_diameter / 2) * 2  # make gpd even
    print("(!) domain_height_in_D (DpY) is even, gridpoints_per_diameter will be " + str(
        gridpoints_per_diameter) + ". Use odd domain_height_in_D (DpY) to enable use of odd GPD!")

# OVERWRITE n_steps, if t_target is given
T_target = 140
if args["t_target"] > 0:
    T_target = args["t_target"]
    n_steps = int(T_target * ((gridpoints_per_diameter) / setup_diameter) * (flow_velocity / (Ma * 1 / np.sqrt(3))))

# SIMULATOR settings
u_init = 0  # initial velocity field: # 0: uniform u=0, # 1: uniform u=1, # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)
perturb_init = True  # perturb initial symmetry by small sine-wave in initial velocity field -> triggers Karman-vortex street for Re > 46
lateral_walls = args["lateral_walls"]
bc_type = args["bc_type"]
vtk_fps = 10  # FramesPerSecond (PU) for vtk-output
cuda_device = args["device"]
nan_reporter = args["nan_reporter"]

gridpoints = gridpoints_per_diameter ** 3 * domain_length_in_D * domain_height_in_D * domain_width_in_D  # calc. total number of gridpoints

##################################################
# DATA OUTPUT SETTINGS (observables, stats and vtk)

output_data = True  # output/log parameters, observables and vtk or vti (if output_vtk=True)
if args["output_vtk"]:
    output_vtk = True  # is overwritten by output_data=False (see below)
else:
    output_vtk = False

from_cpt=False
if args["from_cpt"]:  # if sim should start from checkpoints file
    from_cpt=True
    cpt_file = args["cpt_file"]
    cpt_sim_i = int(args["sim_i"])
write_cpt=False
if args["write_cpt"]:
    write_cpt=True

if output_data:  # only calculate u-profiles if data should be saved
    calculate_velocity_profile = args["calcUProfiles"]
    if calculate_velocity_profile:  # only output u-profiles, if they are calculated
        output_velocity_profile = args["outputUProfiles"]
    else:
        output_velocity_profile = False
else:
    calculate_velocity_profile = False
    output_velocity_profile = False

# naming: specify name/number and parameters to put in directory- and datafile-names
name = args["name"]

if output_data:  # toggle output of parameters, observables and vti/vtk files
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%y%m%d") + "_" + timestamp.strftime("%H%M%S")

    # specify output directory/path
    # output_path = "/mnt/ScratchHDD1/Max_Scratch/lbm_simulations"  # lokal HBRS
    # output_path = "/home/max/Documents/lbm_simulations"  # lokal Bonn
#    output_path = "/home/mbille3s/02_lbm_simulations"  # cluster HBRS
#    scratch_dir = "/scratch/mbille3s/21_LBM/01_data"  # cluster HBRS for vti-output of big simulations
    dir_name = "/data_" + str(timestamp) + "_" + name  # create directory name for all outputs to be saved in
    os.makedirs(output_path + dir_name)

    # specify input directory/path for rerefence data
    # diIlio_path = '/home/mbille/lettuce/myTest/DiIlio_Fig09_data/'  # likal HBRS
#    diIlio_path = '/scratch/mbille3s/21_LBM/03_reference_data/DiIlio_2018/'  # cluster HBRS

    if calculate_velocity_profile:
        os.makedirs(output_path + dir_name + "/AvgVelocity_Data")

    print("dir_name: " + dir_name)
    if output_vtk:
        # vtk_path = output_path+dir_name+"/vtk/out"  # subdirectory for vtk/vti output
        os.makedirs(scratch_dir + dir_name + "/vtk/")
        vtk_path = scratch_dir + dir_name + "/vtk/out"
        print("vtk_path: " + vtk_path)
    elif write_cpt:
        os.makedirs(scratch_dir + dir_name)
else:  # "no output" suppresses the vtk output too
    output_vtk = False

##################################################
# SIM SETUP (instatiate objects, calculate&place obstacle, append reporters)

# lattice (parse stencil choice)
if args["stencil"] == "D3Q27":
    stencil = lt.D3Q27
    stencil_choice = "D3Q27"
elif args["stencil"] == "D3Q19":
    stencil = lt.D3Q19
    stencil_choice = "D3Q19"
else:  # stencil == D3Q15
    stencil = lt.D3Q15
    stencil_choice = "D3Q15"

lattice = lt.Lattice(stencil, cuda_device, dtype=torch.float64)
if args["EQLM"]:
    lattice.equilibrium = QuadraticEquilibrium_LessMemory(lattice)
    print("using QuadraticEquilibrium_LessMemory")

# flow
flow = ObstacleCylinder(shape=(domain_length_in_D * gridpoints_per_diameter,
                               domain_height_in_D * gridpoints_per_diameter,
                               domain_width_in_D * gridpoints_per_diameter),
                        reynolds_number=re, mach_number=Ma,
                        lattice=lattice,
                        char_length_pu=setup_diameter,
                        char_length_lu=gridpoints_per_diameter,
                        char_velocity_pu=flow_velocity,
                        lateral_walls=args["lateral_walls"],
                        bc_type=args["bc_type"],
                        perturb_init=perturb_init,
                        u_init=u_init
                        )

### Simulation-Object (Simulator/solver) and additional settings (tau, collision operator)
tau = flow.units.relaxation_parameter_lu
re_g = flow.units.characteristic_velocity_lu / (
            lattice.stencil.cs ** 2 * (tau - 0.5))  # grid reynolds number (should be O(10))
print("Re_g = ", re_g)

# collision operator
if args["collision"] == "kbc":
    collision = lt.KBCCollision3D(lattice, tau)
    collision_choice = "kbc"
elif args["collision"] == "reg":
    collision = lt.RegularizedCollision(lattice, tau)
    collision_choice = "reg"
else:
    collision = lt.BGKCollision(lattice, tau)
    collision_choice = "bgk"

# solver
sim = Simulation(flow, lattice,
                    collision,
                    # lt.BGKCollision(lattice, tau),
                    # lt.RegularizedCollision(lattice, tau),
                    # lt.KBCCollision2D(lattice,tau),
                    lt.StandardStreaming(lattice)
                    )
### Reporter

# VTK Reporter -> visualization
if output_vtk == True:
    VTKreport = lt.VTKReporter(lattice, flow, interval=int(flow.units.convert_time_to_lu(1 / vtk_fps)),
                               filename_base=vtk_path)
    sim.reporters.append(VTKreport)
    # export obstacle
    mask_dict = dict()
    mask_dict["mask"] = flow.obstacle_mask.astype(int)
    imageToVTK(
        path=scratch_dir + dir_name + "/vtk/obstacle_point",
        pointData=mask_dict
    )
    imageToVTK(
        path=scratch_dir + dir_name + "/vtk/obstacle_cell",
        cellData=mask_dict
    )

# Observable reporter: drag coefficient
DragObservable = DragCoefficient(lattice, flow, sim._boundaries[-1],
                                    area=setup_diameter * flow.units.convert_length_to_pu(
                                        gridpoints_per_diameter * domain_width_in_D))  # create observable // ! area A=2*r is in PU
Dragreport = lt.ObservableReporter(DragObservable, out=None)  # create reporter and link to created observable
sim.reporters.append(Dragreport)  # append reporter to reporter-list of simulator/solver

# Observable reporter: lift coefficient
LiftObservable = LiftCoefficient(lattice, flow, sim._boundaries[-1],
                                    area=setup_diameter * flow.units.convert_length_to_pu(
                                        gridpoints_per_diameter * domain_width_in_D))
Liftreport = lt.ObservableReporter(LiftObservable, out=None)
sim.reporters.append(Liftreport)

# AvgVelocityReporter
if calculate_velocity_profile:
    # define positions
    position_1 = flow.x_pos - 0.5 + 1.06 * flow.radius * 2  # int(round(flow.x_pos + 1.06 * flow.radius * 2 , 0))
    position_2 = flow.x_pos - 0.5 + 1.54 * flow.radius * 2  # int(round(flow.x_pos + 1.54 * flow.radius * 2 , 0))
    position_3 = flow.x_pos - 0.5 + 2.02 * flow.radius * 2  # int(round(flow.x_pos + 2.02 * flow.radius * 2 , 0))
    print("V_avg positions:" + " p1: " + str(position_1) + " p2:  " + str(position_2) + " p3:  " + str(position_3))

    # create and append AvgVelocity-reporter
    AvgVelocity1 = AverageVelocityReporter(lattice, flow, position_1)
    sim.reporters.append(AvgVelocity1)
    AvgVelocity2 = AverageVelocityReporter(lattice, flow, position_2)
    sim.reporters.append(AvgVelocity2)
    AvgVelocity3 = AverageVelocityReporter(lattice, flow, position_3)
    sim.reporters.append(AvgVelocity3)

# NaN STOP
if nan_reporter:
    NaNReporter = NaNReporter(flow, lattice, n_steps, T_target)
    sim.reporters.append(NaNReporter)

##################################################
# PRINT PARAMETERS prior to simulation:
print("shape_LU:", gridpoints_per_diameter * domain_length_in_D, "x", gridpoints_per_diameter * domain_height_in_D, "x",
      gridpoints_per_diameter * domain_width_in_D)
print("T with", n_steps, "steps:",
      round(n_steps * (setup_diameter / (gridpoints_per_diameter)) * (Ma * 1 / np.sqrt(3) / flow_velocity), 2),
      "seconds")
print("n_steps to simulate 1 second:",
      round(((gridpoints_per_diameter) / setup_diameter) * (flow_velocity / (Ma * 1 / np.sqrt(3))), 2), "steps")
print("n_steps to simulate", T_target, "seconds:",
      T_target * round(((gridpoints_per_diameter) / setup_diameter) * (flow_velocity / (Ma * 1 / np.sqrt(3))), 2),
      "steps")
if output_vtk:
    print("generates approx.", int(vtk_fps * (
                n_steps * (setup_diameter / (gridpoints_per_diameter)) * (Ma * 1 / np.sqrt(3) / flow_velocity))) + 1,
          ".vti/.vtk-frames")

##################################################
### export parameters to file

if output_data:
    output_file = open(output_path + dir_name + "/" + timestamp + "_parameters.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-Parameters   ###")
    output_file.write("\nRe = " + str(re))
    output_file.write("\nn_steps = " + str(n_steps))
    output_file.write("\nT_target = " + str(flow.units.convert_time_to_pu(n_steps)) + " seconds")
    output_file.write("\ngridpoints_per_diameter (gpd) = " + str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: " + str(gpd_setup) + " to " + str(
            gridpoints_per_diameter) + " because D/Y is even")
    output_file.write("\nDpX (D/X) = " + str(domain_length_in_D))
    output_file.write("\nDpY (D/Y) = " + str(domain_height_in_D))
    if lattice.D == 3:
        output_file.write("\nDpZ (D/Z) = " + str(domain_width_in_D))
    output_file.write("\nshape_LU: " + str(flow.shape))
    output_file.write(("\ntotal_number_of_gridpoints: " + str(lattice.rho(sim.f).numel())))
    output_file.write("\nbc_type = " + str(bc_type))
    output_file.write("\nlateral_walls = " + str(lateral_walls))
    output_file.write("\nstencil = " + str(stencil_choice))
    output_file.write("\ncollision = " + str(collision_choice))
    output_file.write("\n")
    output_file.write("\nMa = " + str(Ma))
    output_file.write("\ntau = " + str(tau))
    output_file.write("\ngrid_reynolds_number (Re_g) = " + str(re_g))
    output_file.write("\n")
    output_file.write("\nsetup_diameter_PU = " + str(setup_diameter))
    output_file.write("\nflow_velocity_PU = " + str(flow_velocity))
    output_file.write("\nu_init = " + str(u_init))
    output_file.write("\nperturb_init = " + str(perturb_init))
    output_file.write("\n")
    output_file.write("\noutput_vtk = " + str(output_vtk))
    output_file.write("\nvtk_fps = " + str(vtk_fps))
    output_file.close()

### calculate and export 2D obstacle_mask as .png
draw_circular_mask(lattice, gridpoints_per_diameter, output_data=output_data, filebase=output_path+dir_name, print_data=False)

##################################################
# LOAD CHECKPOINT and ajust sim.i
if from_cpt:
    if cpt_file is not None:
        sim.load_checkpoint(cpt_file, device=lattice.device)
        #sim.f = sim.f.to(lattice.device, copy=False)  # if device is different to device used on previous simulation!
        sim.i=cpt_sim_i

##################################################
# RUN SIMULATION

t_start = time.time()

mlups = sim.step(int(n_steps))  # Simulation with n_steps steps

t_end = time.time()
runtime = t_end - t_start
# output stats
print("MLUPS:", mlups)
print("PU-Time: ", flow.units.convert_time_to_pu(n_steps), " seconds")
print("number of steps:", n_steps)
print("runtime: ", runtime, "seconds (", round(runtime / 60, 2), "minutes )")

c_time = sim.time_avg["time_collision"]
fc_time = sim.time_avg["time_store_f_collided"]
s_time = sim.time_avg["time_streaming"]
b_time = sim.time_avg["time_boundary"]
r_time = sim.time_avg["time_reporter"]
sum_time = sim.time_avg["time_collision"] + sim.time_avg["time_store_f_collided"] + sim.time_avg["time_streaming"] + sim.time_avg["time_boundary"] + \
           sim.time_avg["time_reporter"]

print("collision avg. time:", sim.time_avg["time_collision"], "seconds (" + str(round(100 * c_time / sum_time, 2)) + " %)")
print("f_c_store avg. time:", sim.time_avg["time_store_f_collided"], "seconds (" + str(round(100 * fc_time/sum_time, 2)) + " %)")
print("streaming avg. time:", sim.time_avg["time_streaming"], "seconds (" + str(round(100 * s_time / sum_time, 2)) + " %)")
print("boundary avg. time:", sim.time_avg["time_boundary"], "seconds (" + str(round(100 * b_time / sum_time, 2)) + " %)")
print("reporter avg. time:", sim.time_avg["time_reporter"], "seconds (" + str(round(100 * r_time / sum_time, 2)) + " %)")

print("current VRAM (MB): ", torch.cuda.memory_allocated(lattice.device) / 1024 / 1024)
print("max. VRAM (MB): ", torch.cuda.max_memory_allocated(lattice.device) / 1024 / 1024)

[cpuLoad1, cpuLoad5, cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU % avg. over last 1 min, 5 min, 15 min; ", round(cpuLoad1, 2), round(cpuLoad5, 2), round(cpuLoad15, 2))

ram = psutil.virtual_memory()
print("current total RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(
    round(ram.total / (1024 * 1024), 2)) + " MB")

### export stats
if output_data:
    output_file = open(output_path + dir_name + "/" + timestamp + "_stats.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = " + str(runtime) + " seconds (=" + str(runtime / 60) + " minutes)")
    output_file.write("\nMLUPS = " + str(mlups))
    output_file.write("\n")
    output_file.write("\navg. Collision-Time [s] = " + str(c_time) + " (" + str(round(100 * c_time / sum_time, 2)) + " %)")
    output_file.write("\navg. store_fcl-Time [s] = " + str(fc_time) + " (" + str(round(100 * fc_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Streaming-Time [s] = " + str(s_time) + " (" + str(round(100 * s_time / sum_time, 2)) + " %)")
    output_file.write("\navg. Boundary-Time  [s] = " + str(b_time) + " (" + str(round(100 * b_time / sum_time, 2)) + " %)")
    output_file.write("\navg. Reporter-Time  [s] = " + str(r_time) + " (" + str(round(100 * r_time / sum_time, 2)) + " %)")
    output_file.write("\n")
    output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(lattice.device) / 1024 / 1024))
    output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(lattice.device) / 1024 / 1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\ntotal current RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(round(ram.total / (1024 * 1024), 2)) + " MB")
    output_file.close()

##################################################
# WRITE CHECKPOINT
if write_cpt:
    try:
        sim.save_checkpoint(scratch_dir+dir_name+"/"+timestamp + "_f_"+str(sim.i)+".cpt", device="cpu")
    except:
        print("(!) writing checkpoint failed...")

##################################################
# CREATE OBSERVABLE-PLOTS & SAVE OBSERVABLE-timeseries

# Avg VELOCITY
if calculate_velocity_profile:
    avg_u_start = 0.5

    # import reference data: (data is: first collumn Y/D, second column u_d/u_char)
    # ux
    p1_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_LS1993.csv', delimiter=';')
    p2_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_LS1993.csv', delimiter=';')
    p3_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_LS1993.csv', delimiter=';')

    p1_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_KM2000.csv', delimiter=';')
    p2_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_KM2000.csv', delimiter=';')
    p3_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_KM2000.csv', delimiter=';')

    p1_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_WR2008.csv', delimiter=';')
    p2_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_WR2008.csv', delimiter=';')
    p3_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_WR2008.csv', delimiter=';')

    p1_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_DI2018.csv', delimiter=';')
    p2_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_DI2018.csv', delimiter=';')
    p3_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_DI2018.csv', delimiter=';')

    # uy
    p1_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_LS1993.csv', delimiter=';')
    p2_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_LS1993.csv', delimiter=';')
    p3_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_LS1993.csv', delimiter=';')

    p1_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_KM2000.csv', delimiter=';')
    p2_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_KM2000.csv', delimiter=';')
    p3_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_KM2000.csv', delimiter=';')

    p1_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_WR2008.csv', delimiter=';')
    p2_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_WR2008.csv', delimiter=';')
    p3_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_WR2008.csv', delimiter=';')

    p1_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_DI2018.csv', delimiter=';')
    p2_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_DI2018.csv', delimiter=';')
    p3_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_DI2018.csv', delimiter=';')

    # uxux
    p1_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_DI2018.csv', delimiter=';')
    p1_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_KM2000.csv', delimiter=';')
    p1_R2016_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_R2016.csv', delimiter=';')
    p2_BM1994_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_BM1994.csv', delimiter=';')
    p2_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_DI2018.csv', delimiter=';')
    p2_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_KM2000.csv', delimiter=';')
    p2_LS1993_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_LS1993.csv', delimiter=';')
    p2_R2016_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_R2016.csv', delimiter=';')
    p3_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_DI2018.csv', delimiter=';')
    p3_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_KM2000.csv', delimiter=';')
    p3_R2016_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_R2016.csv', delimiter=';')

    # uyuy
    p1_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos1_DI2018.csv', delimiter=';')
    p1_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos1_R2016.csv', delimiter=';')
    p2_BM1994_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_BM1994.csv', delimiter=';')
    p2_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_DI2018.csv', delimiter=';')
    p2_LS1993_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_LS1993.csv', delimiter=';')
    p2_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_R2016.csv', delimiter=';')
    p3_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos3_DI2018.csv', delimiter=';')
    p3_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos3_R2016.csv', delimiter=';')

    # uxuy
    p1_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_BM1994.csv', delimiter=';')
    p1_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_DI2018.csv', delimiter=';')
    p1_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_R2016.csv', delimiter=';')
    p2_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_BM1994.csv', delimiter=';')
    p2_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_DI2018.csv', delimiter=';')
    p2_LS1993_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_LS1993.csv', delimiter=';')
    p2_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_R2016.csv', delimiter=';')
    p3_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_BM1994.csv', delimiter=';')
    p3_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_DI2018.csv', delimiter=';')
    p3_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_R2016.csv', delimiter=';')

    # output sim data to files (not averaged over time)
    if output_data and output_velocity_profile:
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_1_timeseries.npy",
                np.array(AvgVelocity1.out))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_2_timeseries.npy",
                np.array(AvgVelocity2.out))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_3_timeseries.npy",
                np.array(AvgVelocity3.out))

    u1 = np.array(AvgVelocity1.out)[int(avg_u_start * np.array(AvgVelocity1.out).shape[0] - 1):]
    u2 = np.array(AvgVelocity2.out)[int(avg_u_start * np.array(AvgVelocity2.out).shape[0] - 1):]
    u3 = np.array(AvgVelocity3.out)[int(avg_u_start * np.array(AvgVelocity3.out).shape[0] - 1):]

    avg_u1 = np.mean(u1, axis=0)  # time average
    avg_u2 = np.mean(u2, axis=0)  # time average
    avg_u3 = np.mean(u3, axis=0)  # time average

    if output_data:  # output (time-mean) velocity profiles
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_1_t-avg.npy", avg_u1)
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_2_t-avg.npy", avg_u2)
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_3_t-avg.npy", avg_u3)

    avg_u1_x = avg_u1[0]  # u_x component over y at pos 1
    avg_u2_x = avg_u2[0]  # u_x component over y at pos 2
    avg_u3_x = avg_u3[0]  # u_x component over y at pos 3

    avg_u1_y = avg_u1[1]  # u_y component over y at pos 1
    avg_u2_y = avg_u2[1]  # u_y component over y at pos 2
    avg_u3_y = avg_u3[1]  # u_y component over y at pos 3

    y_in_D = (np.arange(avg_u1_x.shape[0]) + 1 - flow.y_pos) / flow.units.characteristic_length_lu  # y/D for figure
    if output_data:
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_YinD.npy", y_in_D)
    cm = 1 / 2.54
    # PLOT ux
    fig, (ax_ux, ax_uy) = plt.subplots(1, 2, constrained_layout=True, figsize=(30 * cm, 10 * cm))
    ax_ux.plot(y_in_D, avg_u1_x, y_in_D, avg_u2_x, y_in_D, avg_u3_x)
    ax_ux.set_xlabel("y/D")
    ax_ux.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
    ax_ux.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

    # OPT. TO DO: add secondary axis for LU-grid
    # ...needs 'function' to convert from y/D in LU and LU in y/D

    # OPT. TO DO: make folder for AvgVelocity-stuff
    # if output_data:
    #     plt.savefig(output_path+dir_name+"/AvgVelocity_x.png")
    # plt.close()

    # PLOT uy
    # fig, ax = plt.subplots(constrained_layout=True)
    ax_uy.plot(y_in_D, avg_u1_y, y_in_D, avg_u2_y, y_in_D, avg_u3_y)
    ax_uy.set_xlabel("y/D")
    ax_uy.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
    ax_uy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

    # OPT. TO DO: add secondary axis for LU-grid
    # ...needs 'function' to convert from y/D in LU and LU in y/D
    # OPT. TO DO: make folder for AvgVelocity-stuff
    # !!! QUESTION: is x/D the position measured FROM the cylinder (x_pos), or measured from x=0 ?

    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_velocity_noReference.png")
    plt.close()

    # PLOT ux against references
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D, avg_u1_x, y_in_D, avg_u2_x - 1, y_in_D, avg_u3_x - 2)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_LS = ax.plot(p1_LS1993_ux[:, 0], p1_LS1993_ux[:, 1], p2_LS1993_ux[:, 0], p2_LS1993_ux[:, 1], p3_LS1993_ux[:, 0],
                     p3_LS1993_ux[:, 1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_KM = ax.plot(p1_KM2000_ux[:, 0], p1_KM2000_ux[:, 1], p2_KM2000_ux[:, 0], p2_KM2000_ux[:, 1], p3_KM2000_ux[:, 0],
                     p3_KM2000_ux[:, 1])
    plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k", label="Kravchenko & Moin (2000)")
    ref_WR = ax.plot(p1_WR2008_ux[:, 0], p1_WR2008_ux[:, 1], p2_WR2008_ux[:, 0], p2_WR2008_ux[:, 1], p3_WR2008_ux[:, 0],
                     p3_WR2008_ux[:, 1])
    plt.setp(ref_WR, ls="dashdot", lw=1.5, marker="", color="k", label="Wissink & Rodi (2008)")
    ref_DI = ax.plot(p1_DI2018_ux[:, 0], p1_DI2018_ux[:, 1], p2_DI2018_ux[:, 0], p2_DI2018_ux[:, 1], p3_DI2018_ux[:, 0],
                     p3_DI2018_ux[:, 1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
    ax.set_ylim([-2.5, +2])
    ax.set_xlim([-3, 3])
    ax.legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0], ref_DI[0]], loc='best')
    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_ux_withReference.png")
    plt.close()

    # PLOT uy against references
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D, avg_u1_y, y_in_D, avg_u2_y - 1, y_in_D, avg_u3_y - 2)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_LS = ax.plot(p1_LS1993_uy[:, 0], p1_LS1993_uy[:, 1], p2_LS1993_uy[:, 0], p2_LS1993_uy[:, 1], p3_LS1993_uy[:, 0],
                     p3_LS1993_uy[:, 1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_KM = ax.plot(p1_KM2000_uy[:, 0], p1_KM2000_uy[:, 1], p2_KM2000_uy[:, 0], p2_KM2000_uy[:, 1], p3_KM2000_uy[:, 0],
                     p3_KM2000_uy[:, 1])
    plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k", label="Kravchenko & Moin (2000)")
    ref_WR = ax.plot(p1_WR2008_uy[:, 0], p1_WR2008_uy[:, 1], p2_WR2008_uy[:, 0], p2_WR2008_uy[:, 1], p3_WR2008_uy[:, 0],
                     p3_WR2008_uy[:, 1])
    plt.setp(ref_WR, ls="dashdot", lw=1.5, marker="", color="k", label="Wissink & Rodi (2008)")
    ref_DI = ax.plot(p1_DI2018_uy[:, 0], p1_DI2018_uy[:, 1], p2_DI2018_uy[:, 0], p2_DI2018_uy[:, 1], p3_DI2018_uy[:, 0],
                     p3_DI2018_uy[:, 1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
    ax.set_ylim([-2.5, +1.5])
    ax.set_xlim([-3, 3])
    ax.legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0], ref_DI[0]], loc='best')
    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_uy_withReference.png")
    plt.close()

    ## turbulent Reynolds stresses

    # diff between timeseries and time_average -> u'
    u1_diff = u1 - avg_u1
    u2_diff = u2 - avg_u2
    u3_diff = u3 - avg_u3

    # square of diff -> u'^2
    u1_diff_sq = u1_diff ** 2
    u2_diff_sq = u2_diff ** 2
    u3_diff_sq = u3_diff ** 2

    # ux'*uy'
    u1_diff_xy = u1_diff[:, 0, :] * u1_diff[:, 1, :]
    u2_diff_xy = u2_diff[:, 0, :] * u2_diff[:, 1, :]
    u3_diff_xy = u3_diff[:, 0, :] * u3_diff[:, 1, :]

    # time_average of u'² and ux'uy'
    u1_diff_sq_mean = np.mean(u1_diff_sq, axis=0)  # time average
    u2_diff_sq_mean = np.mean(u2_diff_sq, axis=0)  # time average
    u3_diff_sq_mean = np.mean(u3_diff_sq, axis=0)  # time average
    u1_diff_xy_mean = np.mean(u1_diff_xy, axis=0)  # time average
    u2_diff_xy_mean = np.mean(u2_diff_xy, axis=0)  # time average
    u3_diff_xy_mean = np.mean(u3_diff_xy, axis=0)  # time average

    if output_data:  # save reynolds stresses
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_1_ReStress_x.npy",
                np.array([y_in_D, u1_diff_sq_mean[0]]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_2_ReStress_x.npy",
                np.array([y_in_D, u2_diff_sq_mean[0]]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_3_ReStress_x.npy",
                np.array([y_in_D, u3_diff_sq_mean[0]]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_1_ReStress_y.npy",
                np.array([y_in_D, u1_diff_sq_mean[1]]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_2_ReStress_y.npy",
                np.array([y_in_D, u2_diff_sq_mean[1]]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_3_ReStress_y.npy",
                np.array([y_in_D, u3_diff_sq_mean[1]]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_1_ReShearStress.npy",
                np.array([y_in_D, u1_diff_xy_mean]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_2_ReShearStress.npy",
                np.array([y_in_D, u2_diff_xy_mean]))
        np.save(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_3_ReShearStress.npy",
                np.array([y_in_D, u3_diff_xy_mean]))

    fig, (ax_xx, ax_yy, ax_xy) = plt.subplots(1, 3, figsize=(40 * cm, 10 * cm), constrained_layout=True)
    ax_xx.plot(y_in_D, u1_diff_sq_mean[0], y_in_D, u2_diff_sq_mean[0], y_in_D, u3_diff_sq_mean[0])
    ax_xx.set_xlabel("y/D")
    ax_xx.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
    ax_xx.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    # if output_data:
    #     plt.savefig(output_path+dir_name+"/AvgVelocity_uxux.png")
    # plt.close()

    # fig, ax = plt.subplots(constrained_layout=True)
    ax_yy.plot(y_in_D, u1_diff_sq_mean[1], y_in_D, u2_diff_sq_mean[1], y_in_D, u3_diff_sq_mean[1])
    ax_yy.set_xlabel("y/D")
    ax_yy.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
    ax_yy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    # if output_data:
    #     plt.savefig(output_path+dir_name+"/AvgVelocity_uyuy.png")
    # plt.close()

    # fig, ax = plt.subplots(constrained_layout=True)
    ax_xy.plot(y_in_D, u1_diff_xy_mean, y_in_D, u2_diff_xy_mean, y_in_D, u3_diff_xy_mean)
    ax_xy.set_xlabel("y/D")
    ax_xy.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
    ax_xy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])

    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_reynoldsStresses_noReference.png")
    plt.close()

    # plot reynolds stresses against reference
    # uxux - streamwise
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D, u1_diff_sq_mean[0], y_in_D, u2_diff_sq_mean[0] - 0.5, y_in_D, u3_diff_sq_mean[0] - 1)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_LS = ax.plot(p2_LS1993_uxux[:, 0], p2_LS1993_uxux[:, 1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_R = ax.plot(p1_R2016_uxux[:, 0], p1_R2016_uxux[:, 1], p3_R2016_uxux[:, 0], p3_R2016_uxux[:, 1],
                    p3_R2016_uxux[:, 0], p3_R2016_uxux[:, 1])
    plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k", label="Rajani et al. (2016)")
    ref_KM = ax.plot(p1_KM2000_uxux[:, 0], p1_KM2000_uxux[:, 1], p2_KM2000_uxux[:, 0], p2_KM2000_uxux[:, 1],
                     p3_KM2000_uxux[:, 0], p3_KM2000_uxux[:, 1])
    plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k", label="Kravchenko & Moin (2000)")
    ref_BM = ax.plot(p2_BM1994_uxux[:, 0], p2_BM1994_uxux[:, 1])
    plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k", label="Beaudan & Moin (1994)")
    ref_DI = ax.plot(p1_DI2018_uxux[:, 0], p1_DI2018_uxux[:, 1], p2_DI2018_uxux[:, 0], p2_DI2018_uxux[:, 1],
                     p3_DI2018_uxux[:, 0], p3_DI2018_uxux[:, 1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
    ax.set_ylim([-1.2, 0.8])
    ax.set_xlim([-3, 3])
    ax.legend(handles=[my_data[0], ref_LS[0], ref_R[0], ref_KM[0], ref_BM[0], ref_DI[0]], loc='best')

    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_uxux_withReference.png")
    plt.close()

    # uyuy - cross-stream
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D, u1_diff_sq_mean[1], y_in_D, u2_diff_sq_mean[1] - 0.5, y_in_D, u3_diff_sq_mean[1] - 1)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_BM = ax.plot(p2_BM1994_uyuy[:, 0], p2_BM1994_uyuy[:, 1])
    plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k", label="Beaudan & Moin (1994)")
    ref_LS = ax.plot(p2_LS1993_uyuy[:, 0], p2_LS1993_uyuy[:, 1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_R = ax.plot(p1_R2016_uyuy[:, 0], p1_R2016_uyuy[:, 1], p3_R2016_uyuy[:, 0], p3_R2016_uyuy[:, 1],
                    p3_R2016_uyuy[:, 0], p3_R2016_uyuy[:, 1])
    plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k", label="Rajani et al. (2016)")
    ref_DI = ax.plot(p1_DI2018_uyuy[:, 0], p1_DI2018_uyuy[:, 1], p2_DI2018_uyuy[:, 0], p2_DI2018_uyuy[:, 1],
                     p3_DI2018_uyuy[:, 0], p3_DI2018_uyuy[:, 1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
    ax.set_ylim([-1.2, 0.8])
    ax.set_xlim([-3, 3])
    ax.legend(handles=[my_data[0], ref_BM[0], ref_LS[0], ref_R[0], ref_DI[0]], loc='best')

    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_uyuy_withReference.png")
    plt.close()

    # uxuy - Reynolds shear stress
    fig, ax = plt.subplots(constrained_layout=True)
    my_data = ax.plot(y_in_D, u1_diff_xy_mean, y_in_D, u2_diff_xy_mean - 0.5, y_in_D, u3_diff_xy_mean - 1)
    plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
    ref_BM = ax.plot(p2_BM1994_uxuy[:, 0], p2_BM1994_uxuy[:, 1])
    plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k", label="Beaudan & Moin (1994)")
    ref_LS = ax.plot(p2_LS1993_uxuy[:, 0], p2_LS1993_uxuy[:, 1])
    plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
    ref_R = ax.plot(p1_R2016_uxuy[:, 0], p1_R2016_uxuy[:, 1], p3_R2016_uxuy[:, 0], p3_R2016_uxuy[:, 1],
                    p3_R2016_uxuy[:, 0], p3_R2016_uxuy[:, 1])
    plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k", label="Rajani et al. (2016)")
    ref_DI = ax.plot(p1_DI2018_uxuy[:, 0], p1_DI2018_uxuy[:, 1], p2_DI2018_uxuy[:, 0], p2_DI2018_uxuy[:, 1],
                     p3_DI2018_uxuy[:, 0], p3_DI2018_uxuy[:, 1])
    plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue", label="Di Ilio et al. (2018)")
    ax.set_xlabel("y/D")
    ax.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
    ax.set_ylim([-1.2, 0.8])
    ax.set_xlim([-3, 3])
    ax.legend(handles=[my_data[0], ref_BM[0], ref_LS[0], ref_R[0], ref_DI[0]], loc='best')

    if output_data:
        plt.savefig(output_path + dir_name + "/AvgVelocity_Data" + "/AvgVelocity_uxuy_withReference.png")
    plt.close()

    # (!) standard plot/figure size in python is 6.4 x 4.8 inches

# DRAG COEFFICIENT
try:
    try:
        drag_coefficient = np.array(Dragreport.out)
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(drag_coefficient[:, 1], drag_coefficient[:, 2])
        ax.set_xlabel("physical time / s")
        ax.set_ylabel("Coefficient of Drag Cd")
        ax.set_ylim([0.5, 1.6])  # change y-limits
        secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
        secax.set_xlabel("timesteps (simulation time / LU)")
    except:
        print("(!) drag_plotting didn't work...'")

    if output_data:
        try:
            plt.savefig(output_path + dir_name + "/drag_coefficient.png")
        except:
            print("(!) saving 'drag_coefficient.png' failed!")
        try:
            np.savetxt(output_path + dir_name + "/drag_coefficient.txt", drag_coefficient,
                       header="stepLU  |  timePU  |  Cd  FROM str(timestamp)")
        except:
            print("(!) saving drag_timeseries failed!")
    ax.set_ylim([drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].min() * 0.5,
                 drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].max() * 1.2])
    if output_data:
        try:
            plt.savefig(output_path + dir_name + "/drag_coefficient_adjusted.png")
        except:
            print("(!) saving drag_coefficient_adjusted.png failed!")
    plt.close()
except:
    print("(!) analysing drag_coefficient didn't work'")

# peak finder: try calculating the mean drag coefficient from an integer number of periods, if a clear periodic signal is found
try:
    values = drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2]

    peaks_max = find_peaks(values, prominence=((values.max() - values.min()) / 2))
    peaks_min = find_peaks(-values, prominence=((values.max() - values.min()) / 2))
    if peaks_min[0].shape[0] - peaks_max[0].shape[0] > 0:
        peak_number = peaks_max[0].shape[0]
    else:
        peak_number = peaks_min[0].shape[0]

    if peaks_min[0][0] < peaks_max[0][0]:
        first_peak = peaks_min[0][0]
        last_peak = peaks_max[0][peak_number - 1]
    else:
        first_peak = peaks_max[0][0]
        last_peak = peaks_min[0][peak_number - 1]

    drag_mean = values[first_peak:last_peak].mean()
    drag_mean_simple = values.mean()

    print("Cd, simple mean:     ", drag_mean_simple)
    print("Cd, peak_finder mean:", drag_mean)

    drag_stepsLU = drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 0]
    peak_max_y = values[peaks_max[0]]
    peak_max_x = drag_stepsLU[peaks_max[0]]
    peak_min_y = values[peaks_min[0]]
    peak_min_x = drag_stepsLU[peaks_min[0]]

    plt.plot(drag_stepsLU, values)
    plt.scatter(peak_max_x[:peak_number], peak_max_y[:peak_number])
    plt.scatter(peak_min_x[:peak_number], peak_min_y[:peak_number])
    plt.scatter(drag_stepsLU[first_peak], values[first_peak])
    plt.scatter(drag_stepsLU[last_peak], values[last_peak])
    plt.savefig(output_path + dir_name + "/drag_coefficient_peakfinder.png")
    peakfinder = True
except:  # if signal is not sinusoidal enough, calculate only simple mean value
    print(
        "peak-finding didn't work... probably no significant peaks visible (Re<46?), or periodic region not reached (T too small)")
    values = drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2]
    drag_mean_simple = values.mean()
    peakfinder = False
    print("Cd, simple mean:", drag_mean_simple)
try:
    plt.close()
except:
    pass

# LIFT COEFFICIENT
try:
    lift_coefficient = np.array(Liftreport.out)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(lift_coefficient[:, 1], lift_coefficient[:, 2])
    ax.set_xlabel("physical time / s")
    ax.set_ylabel("Coefficient of Lift Cl")
    ax.set_ylim([-1.1, 1.1])

    secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
    secax.set_xlabel("timesteps (simulation time / LU)")
    if output_data:
        try:
            plt.savefig(output_path + dir_name + "/lift_coefficient.png")
        except:
            print("(!) saving lift_coefficient.png didn't work!")
        try:
            np.savetxt(output_path + dir_name + "/lift_coefficient.txt", lift_coefficient,
                       header="stepLU  |  timePU  |  Cl  FROM str(timestamp)")
        except:
            print("(!) saving lift_timeline didn't work!")
    Cl_min = lift_coefficient[int(lift_coefficient[:, 2].shape[0] * periodic_start):, 2].min()
    Cl_max = lift_coefficient[int(lift_coefficient[:, 2].shape[0] * periodic_start):, 2].max()
    print("Cl_peaks: \nmin", Cl_min, "\nmax", Cl_max)
    plt.close()
except:
    print("(!) analysing lift_coefficient didn't work!")

# plot DRAG and LIFT together:
try:
    fig, ax = plt.subplots(layout="constrained")
    drag_ax = ax.plot(drag_coefficient[:, 1], drag_coefficient[:, 2], color="tab:blue", label="Drag")
    ax.set_xlabel("physical time / s")
    ax.set_ylabel("Coefficient of Drag Cd")
    ax.set_ylim([0.5, 1.6])

    secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
    secax.set_xlabel("timesteps (simulation time / LU)")

    ax2 = ax.twinx()
    lift_ax = ax2.plot(lift_coefficient[:, 1], lift_coefficient[:, 2], color="tab:orange", label="Lift")
    ax2.set_ylabel("Coefficient of Lift Cl")
    ax2.set_ylim([-1.1, 1.1])

    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)

    if output_data:
        try:
            plt.savefig(output_path + dir_name + "/dragAndLift_coefficient.png")
        except:
            print("(!) saving dragAndLift_coefficient.png didn't work!")
            plt.close()
except:
    print("(!) plotting drag and lift together didn't work!")

# STROUHAL number: (only makes sense for Re>46 and if periodic state is reached)
try:
    ### prototyped fft for frequency detection and calculation of strouhal-number
    # ! Drag_frequency is 2* Strouhal-Freq. Lift-freq. is Strouhal-Freq.

    X = np.fft.fft(lift_coefficient[:, 2])  # fft result (amplitudes)
    N = len(X)  # number of freqs
    n = np.arange(N)  # freq index
    T = N * flow.units.convert_time_to_pu(1)  # total time measured (T_PU)
    freq = n / T  # frequencies (x-axis of spectrum)

    plt.figure
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")  # plot spectrum |X|(f)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.xlim(0, 1)
    # print("max. Amplitude np.abx(X).max():", np.abs(X).max())   # for debugging
    plt.ylim(0, np.abs(X[:int(X.shape[0] * 0.5)]).max())  # ylim, where highes peak is on left half of full spectrum

    if output_data:
        plt.savefig(output_path + dir_name + "/fft_Cl.png")

    freq_res = freq[1] - freq[0]  # frequency-resolution
    X_abs = np.abs(X[:int(X.shape[0] * 0.4)])  # get |X| Amplitude for left half of full spectrum
    freq_peak = freq[np.argmax(X_abs)]  # find frequency with highest amplitude
    print("Frequency Peak:", freq_peak, "+-", freq_res, "Hz")
    # f = Strouhal for St=f*D/U and D=U=1 in PU
except:
    print("fft for Strouhal didn't work")
    freq_res = 0
    freq_peak = 0
plt.close()

##################################################
# OUTPUT DATA and stats to directory

### export VRAM info:

if output_data:
    ### CUDA-VRAM-summary:
    output_file = open(output_path + dir_name + "/" + timestamp + "_GPU_memory_summary.txt", "a")
    output_file.write("DATA for " + timestamp + "\n\n")
    output_file.write(torch.cuda.memory_summary(lattice.device))
    output_file.close()

    try:
        ### list present torch tensors:
        output_file = open(output_path + dir_name + "/" + timestamp + "_GPU_list_of_tensors.txt", "a")
        total_bytes = 0
        import gc

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    output_file.write("\n" + str(obj.size()) + ", " + str(obj.nelement() * obj.element_size()))
                    total_bytes = total_bytes + obj.nelement() * obj.element_size()
            except:
                pass
        # output_file.write("\n\ntotal bytes for tensors:"+str(total_bytes))
        output_file.close()

        ### count occurence of tensors in list of tensors:


        my_file = open(output_path + dir_name + "/" + timestamp + "_GPU_list_of_tensors.txt", "r")
        data = my_file.read()
        my_file.close()
        data_into_list = data.split("\n")
        c = Counter(data_into_list)
        output_file = open(output_path + dir_name + "/" + timestamp + "_GPU_counted_tensors.txt", "a")
        for k, v in c.items():
            output_file.write("type,size,bytes: {}, number: {}\n".format(k, v))
        output_file.write("\ntotal bytes for tensors:" + str(total_bytes))
        output_file.close()
    except:
        print("(!) counting tensors didn't work!")

# output parameters, stats and observables
if output_data:
    output_file = open(output_path + dir_name + "/" + timestamp + "_parms_stats_obs.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-Parameters   ###")
    output_file.write("\nRe = " + str(re))
    output_file.write("\nn_steps = " + str(n_steps))
    output_file.write("\nT_target = " + str(flow.units.convert_time_to_pu(n_steps)) + " seconds")
    output_file.write("\ngridpoints_per_diameter (gpd) = " + str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: " + str(gpd_setup) + " to " + str(
            gridpoints_per_diameter) + " because D/Y is even")
    output_file.write("\nDpX (D/X) = " + str(domain_length_in_D))
    output_file.write("\nDpY (D/Y) = " + str(domain_height_in_D))
    if lattice.D == 3:
        output_file.write("\nDpZ (D/Z) = " + str(domain_width_in_D))
    output_file.write("\nshape_LU: " + str(flow.shape))
    output_file.write(("\ntotal_number_of_gridpoints: " + str(lattice.rho(sim.f).numel())))
    output_file.write("\nbc_type = " + str(bc_type))
    output_file.write("\nlateral_walls = " + str(lateral_walls))
    output_file.write("\nstencil = " + str(stencil_choice))
    output_file.write("\ncollision = " + str(collision_choice))
    output_file.write("\n")
    output_file.write("\nMa = " + str(Ma))
    output_file.write("\ntau = " + str(tau))
    output_file.write("\ngrid_reynolds_number (Re_g) = " + str(re_g))
    output_file.write("\n")
    output_file.write("\nsetup_diameter_PU = " + str(setup_diameter))
    output_file.write("\nflow_velocity_PU = " + str(flow_velocity))
    output_file.write("\nu_init = " + str(u_init))
    output_file.write("\nperturb_init = " + str(perturb_init))
    output_file.write("\n")
    output_file.write("\noutput_vtk = " + str(output_vtk))
    output_file.write("\nvtk_fps = " + str(vtk_fps))

    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = " + str(runtime) + " seconds (=" + str(runtime / 60) + " minutes)")
    output_file.write("\nMLUPS = " + str(mlups))
    output_file.write("\n")
    output_file.write("\navg. Collision-Time [s] = " + str(c_time) + " (" + str(round(100 * c_time / sum_time, 2)) + " %)")
    output_file.write("\navg. store_fcl-Time [s] = " + str(fc_time) + " (" + str(round(100 * fc_time/sum_time, 2)) + " %)")
    output_file.write("\navg. Streaming-Time [s] = " + str(s_time) + " (" + str(round(100 * s_time / sum_time, 2)) + " %)")
    output_file.write("\navg. Boundary-Time  [s] = " + str(b_time) + " (" + str(round(100 * b_time / sum_time, 2)) + " %)")
    output_file.write("\navg. Reporter-Time  [s] = " + str(r_time) + " (" + str(round(100 * r_time / sum_time, 2)) + " %)")
    output_file.write("\n")
    output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(lattice.device) / 1024 / 1024))
    output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(lattice.device) / 1024 / 1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\ntotal current RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(round(ram.total / (1024 * 1024), 2)) + " MB")

    output_file.write("\n\n###   OBSERVABLES   ###")
    output_file.write("\nCoefficient of drag between " + str(round(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1), 1], 2)) + " s and " + str(round(drag_coefficient[int(drag_coefficient.shape[0] - 1), 1], 2)) + " s:")
    output_file.write("\nCd_mean, simple      = " + str(drag_mean_simple))
    if peakfinder:
        output_file.write("\nCd_mean, peak_finder = " + str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write(
        "\nCd_min = " + str(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].min()))
    output_file.write(
        "\nCd_max = " + str(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].max()))
    output_file.write("\n")
    output_file.write("\nCoefficient of lift:")
    output_file.write("\nCl_min = " + str(Cl_min))
    output_file.write("\nCl_max = " + str(Cl_max))
    output_file.write("\n")
    output_file.write("\nStrouhal number:")
    output_file.write("\nSt +- df = " + str(freq_peak) + " +- " + str(freq_res) + " Hz")
    output_file.write("\n")
    output_file.close()

# output copyable numbers for EXCEL etc.
if output_data:
    output_file = open(output_path + dir_name + "/" + timestamp + "_parms_stats_obs_copyable.txt", "a")

    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   Data:   ###")
    output_file.write(
        "\nRe, n_steps, t_target(PU), GPD, DpX, DpY, (DpZ), shape_LU, gridpoints, bc_type, lateral_walls, stencil, collision, Ma, tau, Re_grid, setup_diameter_PU, flow_velocity_PU, u_init, perturb_init, output_vtk, vtk_fps, runtime, MLUPS, c_time, fc_time, s_time, b_time, r_time, VRAM_current_MB, VRAM_peak_MB, periodic_start, Cd_mean, Cd_mean_pf, Cd_min, Cd_max, Cl_min, Cl_max, St, df\n")
    output_file.write("\n" + str(re))
    output_file.write("\n" + str(n_steps))
    output_file.write("\n" + str(flow.units.convert_time_to_pu(n_steps)))
    output_file.write("\n" + str(gridpoints_per_diameter))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: " + str(gpd_setup) + " to " + str(
            gridpoints_per_diameter) + " because D/Y is even")
    output_file.write("\n" + str(domain_length_in_D))
    output_file.write("\n" + str(domain_height_in_D))
    if lattice.D == 3:
        output_file.write("\n" + str(domain_width_in_D))
    output_file.write("\n" + str(flow.shape))
    output_file.write("\n" + str(lattice.rho(sim.f).numel()))
    output_file.write("\n" + str(bc_type))
    output_file.write("\n" + str(lateral_walls))
    output_file.write("\n" + str(stencil_choice))
    output_file.write("\n" + str(collision_choice))
    output_file.write("\n")
    output_file.write("\n" + str(Ma))
    output_file.write("\n" + str(tau))
    output_file.write("\n" + str(re_g))
    output_file.write("\n")
    output_file.write("\n" + str(setup_diameter))
    output_file.write("\n" + str(flow_velocity))
    output_file.write("\n" + str(u_init))
    output_file.write("\n" + str(perturb_init))
    output_file.write("\n")
    output_file.write("\n" + str(output_vtk))
    output_file.write("\n" + str(vtk_fps))
    output_file.write("\n")

    output_file.write("\n" + str(runtime))
    output_file.write("\n" + str(mlups))
    output_file.write("\n")
    output_file.write("\n" + str(c_time))
    output_file.write("\n" + str(fc_time))
    output_file.write("\n" + str(s_time))
    output_file.write("\n" + str(b_time))
    output_file.write("\n" + str(r_time))
    output_file.write("\n")
    output_file.write("\n" + str(torch.cuda.memory_allocated(lattice.device) / 1024 / 1024))
    output_file.write("\n" + str(torch.cuda.max_memory_allocated(lattice.device) / 1024 / 1024))
    output_file.write("\n")

    output_file.write("\n" + str(periodic_start))
    output_file.write("\n" + str(drag_mean_simple))
    if peakfinder:
        output_file.write("\n" + str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write("\n" + str(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].min()))
    output_file.write("\n" + str(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].max()))
    output_file.write("\n")
    output_file.write("\n" + str(Cl_min))
    output_file.write("\n" + str(Cl_max))
    output_file.write("\n")
    output_file.write("\n" + str(freq_peak))
    output_file.write("\n" + str(freq_res))
    output_file.write("\n")
    output_file.close()

