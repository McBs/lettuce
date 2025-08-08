import warnings

import torch
import torch.distributed as dist
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional
from abc import ABC, abstractmethod

from . import *
from .cuda_native import NativeCollision, Generator

__all__ = ['Collision', 'Reporter', 'Simulation']


class Collision(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self) -> 'NativeCollision':
        ...


class Reporter(ABC):
    interval: int

    def __init__(self, interval: int):
        self.interval = interval

    @abstractmethod
    def __call__(self, simulation: 'Simulation'):
        ...


class Simulation:
    flow: 'Flow'
    context: 'Context'
    collision: 'Collision'
    boundaries: List['Boundary']
    no_collision_mask: Optional[torch.Tensor]
    no_streaming_mask: Optional[torch.Tensor]
    reporter: List['Reporter']
    #overlap = 8
    #overlap_counter = 1

    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter'],disrtributed: Optional['disrtributed'] = None):
        self.flow = flow
        self.flow.collision = collision
        self.context = flow.context
        self.collision = collision
        self.reporter = reporter
        self.boundaries = ([None]
                           + sorted(flow.boundaries, key=lambda b: str(b)))
        self.disrtributed = disrtributed

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

        # ============================== #
        # generate cuda_native implementation #
        # ============================== #

        def collide_and_stream(*_, **__):
            self._collide()
            self._stream()

        self._collide_and_stream = collide_and_stream

        if self.context.use_native:

            # check for availability of cuda_native for all components

            if (self.flow.equilibrium is not None
                    and not self.flow.equilibrium.native_available()):
                name = self.flow.equilibrium.__class__.__name__
                print(f"cuda_native was requested, but equilibrium '{name}' "
                      f"does not support cuda_native.")
            if not self.collision.native_available():
                name = self.collision.__class__.__name__
                print(f"cuda_native was requested, but collision '{name}' "
                      f"does not support cuda_native.")
            for boundary in self.boundaries[1:]:
                if not boundary.native_available():
                    name = boundary.__class__.__name__
                    print(f"cuda_native was requested, but boundary '{name}' "
                          f"does not support cuda_native.")

            # create cuda_native equivalents

            native_equilibrium = None
            if self.flow.equilibrium is not None:
                native_equilibrium = self.flow.equilibrium.native_generator()

            native_collision = self.collision.native_generator()

            native_boundaries = []
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                native_boundaries.append(boundary.native_generator(i))

            # begin generating cuda_native module from cuda_native components

            generator = Generator(self.flow.stencil, native_collision,
                                  native_boundaries, native_equilibrium)

            native_kernel = generator.resolve()
            if native_kernel is None:

                buffer = generator.generate()
                directory = generator.format(buffer)
                generator.install(directory)

                native_kernel = generator.resolve()
                if native_kernel is None:
                    print('Failed to install cuda_native Extension!')
                    return

            # redirect collide and stream to cuda_native kernel

            self._collide_and_stream = native_kernel

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
        if self.no_collision_mask is None:
            self.flow.f = self.collision(self.flow)
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                self.flow.f = boundary(self.flow)
        else:
            torch.where(torch.eq(self.no_collision_mask, 0),
                        self.collision(self.flow), self.flow.f,
                        out=self.flow.f)
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                torch.where(torch.eq(self.no_collision_mask, i),
                            boundary(self.flow), self.flow.f, out=self.flow.f)
        return self.flow.f

    def _report(self):
        for reporter in self.reporter:
            reporter(self)
    
    def _exchange_messages(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        left_neighbor  = (rank - 1) % world_size
        right_neighbor = (rank + 1) % world_size
        
        if self.flow.remainder > 0:

            if rank < self.flow.remainder:
                if rank == self.flow.remainder -1:
                    big_small_overlap = self.flow.upperfill_big + self.flow.lowerfill_small
                    send_slice_right = self.flow.f[:,-big_small_overlap,...].cpu().clone().detach()
                    recv_slice_right = torch.empty_like(send_slice_right)
                else:
                    big_split_overlap = self.flow.upperfill_big + self.flow.lowerfill_big
                    send_slice_right = self.flow.f[:,-big_split_overlap,...].cpu().clone().detach()      
                    recv_slice_right = torch.empty_like(send_slice_right)

                if rank == 0:
                    c_origin_overlap = 2 * min(self.flow.upperfill_small, self.flow.lowerfill_big)
                    send_slice_left = self.flow.f[:,c_origin_overlap-1,...].cpu().clone().detach()
                    recv_slice_left = torch.empty_like(send_slice_left)

                else:
                    big_split_overlap = self.flow.upperfill_big + self.flow.lowerfill_big
                    send_slice_left = self.flow.f[:,big_split_overlap-1,...].cpu().clone().detach()
                    recv_slice_left = torch.empty_like(send_slice_left)

            else:
                if rank == world_size - 1:
                    print("Rank: " + str(dist.get_rank()) + " upperfil_big: " + str(self.flow.upperfill_big) + " lowerfill_small: " + str(self.flow.lowerfill_small) + " 2 * min: " +str(2 * min(self.flow.upperfill_big, self.flow.lowerfill_small)))
                    c_origin_overlap = min(self.flow.upperfill_big, self.flow.lowerfill_small)
                    send_slice_right = self.flow.f[:,-c_origin_overlap,...].cpu().clone().detach()
                    recv_slice_right = torch.empty_like(send_slice_right)
                else:
                    small_split_overlap =  self.flow.lowerfill_small + self.flow.upperfill_small
                    send_slice_right = self.flow.f[:,-small_split_overlap,...].cpu().clone().detach()
                    recv_slice_right = torch.empty_like(send_slice_right)

                if rank == self.flow.remainder:
                    big_small_overlap = self.flow.upperfill_big + self.flow.lowerfill_small
                    send_slice_left = self.flow.f[:,big_small_overlap-1,...].cpu().clone().detach()
                    recv_slice_left = torch.empty_like(send_slice_left)


                else:
                    small_split_overlap =  self.flow.lowerfill_small + self.flow.upperfill_small
                    send_slice_left = self.flow.f[:,small_split_overlap-1,...].cpu().clone().detach() 
                    recv_slice_left = torch.empty_like(send_slice_left)

        else:
            

            send_slice_right = self.flow.f[:,-self.flow.overlap,...].cpu().clone().detach()
            send_slice_left = self.flow.f[:,self.flow.overlap-1,...].cpu().clone().detach()

            recv_slice_right = torch.empty_like(send_slice_right)
            recv_slice_left = torch.empty_like(send_slice_left)

        send_req_right = dist.isend(tensor=send_slice_right, dst=right_neighbor)
        recv_req_right = dist.irecv(tensor=recv_slice_right, src=left_neighbor)

        send_req_right.wait()
        recv_req_right.wait()

        send_req_left = dist.isend(tensor=send_slice_left, dst=left_neighbor)
        recv_req_left = dist.irecv(tensor=recv_slice_left, src=right_neighbor)
        
        send_req_left.wait()
        recv_req_left.wait()
       
        self.flow.f[:,0,...]=recv_slice_right.clone().detach()

        self.flow.f[:,-1,...]=recv_slice_left.clone().detach()


    def safe_f(self):
        path = "/home/user/data/"
        if self.disrtributed == "mpi": 
            string = path + "F" + str(self.flow.i) + "_" + str(dist.get_world_size()) + "_nodes_rank_" + str(dist.get_rank()) + "R100Res" + str(self.flow.resolution[0]) + ".pt"  
        else:
            string = path + "F" + str(self.flow.i) + "_Serial_R100Res" + str(self.flow.resolution[0]) + ".pt"
        torch.save(self.flow.f, string)   

    def __call__(self, num_steps):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        

        print("------------Stencil: Dimension: " + str(self.flow.stencil.d) + "---------------")


        for _ in range(num_steps):
            self._collide_and_stream(self)
        #    if overlap_counter == overlap:
        #        print(overlap_counter)
            if self.disrtributed == "mpi":
                self._exchange_messages()
        #        overlap_counter = 1
        #        print(overlap_counter)
        #    else:
        #        print(overlap_counter)
        #        overlap_counter += 1
        #        print(overlap_counter)
            self.flow.i += 1
            self._report()
            if self.flow.i == 1:
                self.safe_f()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)
