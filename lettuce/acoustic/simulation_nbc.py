import warnings

import torch
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional
from abc import ABC, abstractmethod

from . import *

__all__ = ['Reporter', 'SimulationNBC']



class Reporter(ABC):
    interval: int

    def __init__(self, interval: int):
        self.interval = interval

    @abstractmethod
    def __call__(self, simulation: 'Simulation'):
        ...


class SimulationNBC:
    flow: 'Flow'
    context: 'Context'
    collision: 'Collision'
    boundaries: List['Boundary']
    no_collision_mask: Optional[torch.Tensor]
    no_streaming_mask: Optional[torch.Tensor]
    reporter: List['Reporter']

    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter']):
        self.flow = flow
        self.flow.collision = collision
        self.context = flow.context
        self.collision = collision
        self.reporter = reporter
        self.boundaries = ([None]
                           + sorted(flow.boundaries, key=lambda b: str(b)))

        # ==================================== #
        # initialise masks based on boundaries #
        # ==================================== #

        # if there are no boundaries
        # leave the masks uninitialised
        self.no_collision_mask = None
        self.no_streaming_mask = None

        # else initialise the masks
        # based on the boundaries masks
        if len(self.boundaries) > 1:

            self.no_collision_mask = self.context.zero_tensor(
                flow.resolution, dtype=torch.uint8)
            self.no_streaming_mask = self.context.zero_tensor(
                [flow.stencil.q, *flow.resolution], dtype=torch.uint8)

            for i, boundary in enumerate(self.boundaries[1:], start=1):
                ncm = boundary.make_no_collision_mask(
                    [it for it in self.flow.f.shape[1:]], context=self.context)
                if ncm is not None:
                    self.no_collision_mask[ncm] = i
                nsm = boundary.make_no_streaming_mask(
                    [it for it in self.flow.f.shape], context=self.context)
                if nsm is not None:
                    self.no_streaming_mask |= nsm
                self.collision_ids = torch.tensor([0]).to(device=self.no_collision_mask.device)

        # ============================== #
        # generate cuda_native implementation #
        # ============================== #

        def collide_and_stream(*_, **__):
            self._collide()
            self._stream()

        self._collide_and_stream = collide_and_stream


    def step(self, num_steps: int):
        warnings.warn("lt.Simulation.step() is deprecated and will be "
                      "removed in a future version. Instead, call simulation "
                      "directly: simulation(num_steps)", DeprecationWarning)
        return self(num_steps)

    @property
    def units(self):
        return self.flow.units

    @staticmethod
    def __stream(f, i, e, d):
        return torch.roll(f[i], shifts=tuple(e[i]), dims=tuple(np.arange(d)))

    def _stream(self):
        for i in range(1, self.flow.stencil.q):
            if self.no_streaming_mask is None:
                self.flow.f[i] = self.__stream(self.flow.f, i,
                                               self.flow.stencil.e,
                                               self.flow.stencil.d)
            else:
                new_fi = self.__stream(self.flow.f, i, self.flow.stencil.e,
                                       self.flow.stencil.d)
                self.flow.f[i] = torch.where(torch.eq(
                    self.no_streaming_mask[i], 1), self.flow.f[i], new_fi)
        return self.flow.f

    def _collide(self):
        # for i, boundary in enumerate(self.boundaries[1:], start=1):
        #     torch.where(torch.eq(self.no_collision_mask, i),
        #                 boundary(self.flow), self.flow.f, out=self.flow.f)

        for boundary in self.boundaries[1:]:
            self.flow.f = boundary(self.flow)
        self.flow.f = self.collision(self.flow)
        return self.flow.f

    def _report(self):
        for reporter in self.reporter:
            reporter(self)

    def __call__(self, num_steps):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        for _ in range(num_steps):
            self._collide_and_stream(self)
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)
