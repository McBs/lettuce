"""
Streaming Step
"""

import torch
import numpy as np
import lettuce.mpiClass as mpii

__all__ = ["StandardStreaming"]


class StandardStreaming:
    """Standard Streaming step on a regular grid.

    Attributes
    ----------
    no_stream_mask : torch.Tensor
        Boolean mask with the same shape as the distribution function f.
        If None, stream all (also around all boundaries).
    """

    def __init__(self, lattice, mpiObject=None):
        self.lattice = lattice
        self._no_stream_mask = None

        if(mpiObject is not None):
            self.mpiObject=mpiObject
        else:
            self.mpiObject=mpii.mpiObject(0)
        
        if(self.mpiObject.mpi==1):    
            global os
            import os
            global Process
            from torch.multiprocessing import Process
            global dist
            import torch.distributed as dist
            print("mpiStream")

            self.localCall=self.callMPI
            self.size = self.mpiObject.size
            self.rank = self.mpiObject.rank
            self.prev = self.rank - 1 if self.rank != 0 else self.size - 1
            self.next = self.rank + 1 if self.rank != self.size - 1 else 0
            self.forward = np.argwhere(self.lattice.stencil.e[:, 0] > 0)
            self.rest = np.argwhere(self.lattice.stencil.e[:, 0] == 0)
            self.backward = np.argwhere(self.lattice.stencil.e[:, 0] < 0)
            self.no_stream_mask = None
        
        else:
            self.localCall=self.callNonMPI

    @property
    def no_stream_mask(self):
        return self._no_stream_mask

    @no_stream_mask.setter
    def no_stream_mask(self, mask):
        self._no_stream_mask = mask

    def __call__(self, f):
        return self.localCall(f)

    def callNonMPI(self, f):
        for i in range(1, self.lattice.Q):
            if self.no_stream_mask is None:
                f[i] = self._stream(f, i)
            else:
                new_fi = self._stream(f, i)
                f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
        return f

    def _stream(self, f, i):
        return torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))

    def callMPI(self, f):
        cpudevice=torch.device("cpu")
        output_forward = f[self.forward, -1, ...].detach().clone().cpu().contiguous()
         
        output_backward = f[self.backward, 0, ...].detach().clone().cpu().contiguous()
       
        input_forward = torch.zeros_like(output_forward)
        input_backward = torch.zeros_like(output_backward)
        outf = dist.isend(tensor=output_forward, dst=self.next)
        outb = dist.isend(tensor=output_backward, dst=self.prev)
        inf = dist.irecv(tensor=input_forward.contiguous(), src=self.prev)
        inb = dist.irecv(tensor=input_backward.contiguous(), src=self.next)
        
        
        f = torch.cat((torch.zeros_like(f[:, 0, ...]).unsqueeze(1), f, torch.zeros_like(f[:, 0, ...]).unsqueeze(1)), dim=1)
       
        inf.wait()
        #WIP: vor diesem wait schon mal rest streamen?
        input_forwardfinal=self.lattice.convert_to_tensor(input_forward)
        f[self.forward, 0 , ...] = input_forwardfinal
            
        inb.wait()
        input_backwardfinal=self.lattice.convert_to_tensor(input_backward)
        f[self.backward,-1, ...] = input_backwardfinal
        
        outf.wait()
        outb.wait()
        return f

    def stream(self, f):
        if self.no_stream_mask is not None:
            no_stream_mask = torch.cat((torch.zeros_like(self.no_stream_mask[:, 0, ...]),
                                        self.no_stream_mask,
                                        torch.zeros_like(self.no_stream_mask[:, -1, ...])), dim=1)

        for i in range(1, self.lattice.Q):
            i = int(i)
            if self.no_stream_mask is None:
                f[i] = self._stream(f, i)
            else:
                new_fi = self._stream(f, i)
                f[i] = torch.where(no_stream_mask[i], f[i], new_fi)
        return f

    def transmitWholeParts(self,f):
        cpudevice=torch.device("cpu")
        backward = np.argwhere(self.lattice.stencil.e[:, 0] < 99999)

        #Streams everything even if it is not needed
        output_forward = f[backward, -1, ...].detach().clone().cpu().contiguous()
        for i in range(2,3+1):
            output_forwardp=f[backward, -i, ...].detach().clone().cpu().contiguous()
            output_forward=torch.cat((output_forwardp,output_forward),dim=1)
         
        output_backward = f[backward, 0, ...].detach().clone().cpu().contiguous()
        for i in range(1,3):
            output_backwardp=f[backward, i, ...].detach().clone().cpu().contiguous()
            output_backward=torch.cat((output_backward,output_backwardp),dim=1)
        input_forward = torch.zeros_like(output_forward)
        input_backward = torch.zeros_like(output_backward)
        outf = dist.isend(tensor=output_forward, dst=self.next)
        outb = dist.isend(tensor=output_backward, dst=self.prev)
        inf = dist.irecv(tensor=input_forward.contiguous(), src=self.prev)
        inb = dist.irecv(tensor=input_backward.contiguous(), src=self.next)
        selectnmany=[]
        for i in range(0,3):
            selectnmany.append(i)
        f = torch.cat((torch.zeros_like(f[:, selectnmany, ...]), f, torch.zeros_like(f[:, selectnmany, ...])), dim=1)
         
        selectnmanyneg=[]
        for i in range(-3,0):
            selectnmanyneg.append(i)
       
        inf.wait()
        #WIP: vor diesem wait schon mal rest streamen?
        input_forwardfinal=self.lattice.convert_to_tensor(input_forward)
        f[backward, selectnmany , ...] = input_forwardfinal
            
        inb.wait()
        input_backwardfinal=self.lattice.convert_to_tensor(input_backward)
        f[backward,selectnmanyneg, ...] = input_backwardfinal
        outf.wait()
        outb.wait()
        return f

    def transmitWhole5Parts(self,f):
        cpudevice=torch.device("cpu")
        backward = np.argwhere(self.lattice.stencil.e[:, 0] < 99999)

        #Streams everything even if it is not needed
        output_forward = f[backward, -1, ...].detach().clone().cpu().contiguous()
        for i in range(2,5+1):
            output_forwardp=f[backward, -i, ...].detach().clone().cpu().contiguous()
            output_forward=torch.cat((output_forwardp,output_forward),dim=1)

        
         
        output_backward = f[backward, 0, ...].detach().cpu().clone().contiguous()
        for i in range(1,5):
            output_backwardp=f[backward, i, ...].detach().cpu().clone().contiguous()
            output_backward=torch.cat((output_backward,output_backwardp),dim=1)
        input_forward = torch.zeros_like(output_forward,device=cpudevice)
        input_backward = torch.zeros_like(output_backward,device=cpudevice)
        outf = dist.isend(tensor=output_forward, dst=self.next)
        outb = dist.isend(tensor=output_backward, dst=self.prev)
        inf = dist.irecv(tensor=input_forward.contiguous(), src=self.prev)
        inb = dist.irecv(tensor=input_backward.contiguous(), src=self.next)
        selectnmany=[]
        for i in range(0,5):
            selectnmany.append(i)
        f = torch.cat((torch.zeros_like(f[:, selectnmany, ...]), f, torch.zeros_like(f[:, selectnmany, ...])), dim=1)
         
        selectnmanyneg=[]
        for i in range(-5,0):
            selectnmanyneg.append(i)
       
        inf.wait()
        #WIP: vor diesem wait schon mal rest streamen?
        input_forwardfinal=self.lattice.convert_to_tensor(input_forward)
        f[backward, selectnmany , ...] = input_forwardfinal
            
        inb.wait()
        input_backwardfinal=self.lattice.convert_to_tensor(input_backward)
        f[backward,selectnmanyneg, ...] = input_backwardfinal
        outf.wait()
        outb.wait()
        return f

    def reduce(self,f):
        return f[:, 1:-1, ...]
        
    def reduce3(self,f):
        return f[:, 3:-3, ...]
    
    def reduce5(self,f):
        return f[:, 5:-5, ...]

    


class SLStreaming:
    """
    TODO (is there a good python package for octrees or do we have to write this ourselves?)
    """

    def __init__(self, lattice, grid):
        raise NotImplementedError
