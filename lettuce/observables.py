"""
Observables.
Each observable is defined as a callable class.
The `__call__` function takes f as an argument and returns a torch tensor.
"""

import torch
import numpy as np
from torch._C import device
from lettuce.util import torch_gradient
from packaging import version
import lettuce.mpiClass

__all__ = ["Observable", "MaximumVelocity", "IncompressibleKineticEnergy", "Enstrophy", "EnergySpectrum","Mass"]


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow
        

    def __call__(self, f):
        raise NotImplementedError


class MaximumVelocity(Observable):
    """Maximum velocitiy"""
    def __init__(self, lattice, flow,mpiObj=None):
        self.lattice=lattice
        self.flow=flow
        self.mpiObj=mpiObj
        if(self.mpiObj is not None):
            if(mpiObj.mpi==1):
                #we have a distributed array!
                global dist
                import torch.distributed as dist 
                self.calling=self.mpiCall
            
            else:
                self.calling=self.nonMPIcall

        else:
            self.mpiObj=lettuce.mpiClass.mpiObject(0)
            self.calling=self.nonMPIcall

    def nonMPIcall(self,f):
        u = self.lattice.u(f)
        # if mpi: collective max operation of torch.norm(u, dim=0) 
        return self.flow.units.convert_velocity_to_pu(torch.norm(u, dim=0).max())

    def mpiCall(self,f):
        u = self.lattice.u(f)
        normMAX=torch.norm(u, dim=0).max()
        dist.reduce(normMAX,0,dist.ReduceOp.MAX)
        return self.flow.units.convert_velocity_to_pu(normMAX)
        
    def __call__(self, f):
        return self.calling(f)


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""
    def __init__(self, lattice, flow,mpiObj=None):
        self.mpiObj=mpiObj
        self.flow=flow
        self.lattice=lattice
        if(self.mpiObj is not None):
            if(mpiObj.mpi==1):
                #we have a distributed array!
                global dist
                import torch.distributed as dist 
                #set up warnings for that case       
                if(lattice.dtype==torch.float32):
                    print("Warning: Because float addition is not associativ results may differ")
                self.calling=self.mpicall
            
            else:
                self.calling=self.nonMPIcall

        else:
            self.mpiObj=lettuce.mpiClass.mpiObject(0)
            self.calling=self.nonMPIcall

    def __call__(self, f):
        return self.calling(f)
    
    def nonMPIcall(self,f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(torch.sum(self.lattice.incompressible_energy(f)))
        kinE *= dx ** self.lattice.D
        return kinE
    
    def mpicall(self,f):
        dx = self.flow.units.convert_length_to_pu(1.0)
        mpiSum=torch.sum(self.lattice.incompressible_energy(f))
        dist.reduce(mpiSum,0,dist.ReduceOp.SUM)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(mpiSum)
        kinE *= dx ** self.lattice.D
        return kinE


class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """

    def __init__(self, lattice, flow,mpiObj=None):
        self.lattice = lattice
        self.flow = flow
        self.mpiObj=mpiObj
        if(self.mpiObj is not None):
            if(mpiObj.mpi==1):
                #we have a distributed array!
                global dist
                import torch.distributed as dist 
                self.calling=self.mpiCall
                self.next=mpiObj.next
                self.prev=mpiObj.prev
            
            else:
                self.calling=self.nonMPIcall

        else:
            self.mpiObj=lettuce.mpiClass.mpiObject(0)
            self.calling=self.nonMPIcall

    #kopieren der streaming distibuted funktion
    def transmitWholeParts(self,u,num):
        cpudevice=torch.device("cpu")
        if num==3:
            backward = np.array(([[0],[1],[2]]))
        else:
            backward = np.array(([[0],[1]]))
        #Streams everything even if it is not needed
        output_forward = u[backward, -1, ...].detach().clone().cpu().contiguous()
        for i in range(2,3+1):
            output_forwardp=u[backward, -i, ...].detach().clone().cpu().contiguous()
            output_forward=torch.cat((output_forwardp,output_forward),dim=1)

        
         
        output_backward = u[backward, 0, ...].detach().clone().cpu().contiguous()
        for i in range(1,3):
            output_backwardp=u[backward, i, ...].detach().clone().cpu().contiguous()
            output_backward=torch.cat((output_backward,output_backwardp),dim=1)
        input_forward = torch.zeros_like(output_forward,device=cpudevice)
        input_backward = torch.zeros_like(output_backward,device=cpudevice)
        outf = dist.isend(tensor=output_forward, dst=self.next)
        outb = dist.isend(tensor=output_backward, dst=self.prev)
        inf = dist.irecv(tensor=input_forward.contiguous(), src=self.prev)
        inb = dist.irecv(tensor=input_backward.contiguous(), src=self.next)
        selectnmany=[]
        for i in range(0,3):
            selectnmany.append(i)
        u = torch.cat((torch.zeros_like(u[:, selectnmany, ...]), u, torch.zeros_like(u[:, selectnmany, ...])), dim=1)
         
        selectnmanyneg=[]
        for i in range(-3,0):
            selectnmanyneg.append(i)
       
        inf.wait()
        #WIP: vor diesem wait schon mal rest streamen?
        input_forwardfinal=self.lattice.convert_to_tensor(input_forward)
        u[backward, selectnmany , ...] = input_forwardfinal
       
            
        inb.wait()
        input_backwardfinal=self.lattice.convert_to_tensor(input_backward)
        u[backward,selectnmanyneg, ...] = input_backwardfinal
        outf.wait()
        outb.wait()
        return u

    def reduce(self,u):
        return u[:, 3:-3, ...]

    def __call__(self, f):
        return self.calling(f)
    
    def nonMPIcall(self, f):
        #wenn dies ausgeführt wird müssen wir ganze domäne übertragen
        u0 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        if self.lattice.D == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx ** self.lattice.D

    def mpiCall(self, f):
        #wenn dies ausgeführt wird müssen wir ganze domäne übertragen
        if self.lattice.D == 3:
            local_u=self.lattice.u(f)
            local_u=self.transmitWholeParts(local_u,3)

            u0 = self.flow.units.convert_velocity_to_pu(local_u[0])
            u1 = self.flow.units.convert_velocity_to_pu(local_u[1])
            u2 = self.flow.units.convert_velocity_to_pu(local_u[2])
            
            dx = self.flow.units.convert_length_to_pu(1.0)
            grad_u0 = torch_gradient(u0, dx=dx, order=6)
            grad_u1 = torch_gradient(u1, dx=dx, order=6)
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            #passend zurechtschneiden
            grad_u0=self.reduce(grad_u0)
            grad_u1=self.reduce(grad_u1)   
            grad_u2=self.reduce(grad_u2) 

            vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
            dist.reduce(vorticity,0,dist.ReduceOp.SUM)
        else:
            local_u=self.lattice.u(f)
            
            local_u=self.transmitWholeParts(local_u,2)

            u0 = self.flow.units.convert_velocity_to_pu(local_u[0])
            u1 = self.flow.units.convert_velocity_to_pu(local_u[1])
            dx = self.flow.units.convert_length_to_pu(1.0)
            grad_u0 = torch_gradient(u0, dx=dx, order=6)
            grad_u1 = torch_gradient(u1, dx=dx, order=6)
            #passend zurechtschneiden
            grad_u0=self.reduce(grad_u0)
            grad_u1=self.reduce(grad_u1)
            vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]))
        
        return vorticity * dx ** self.lattice.D


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""

    def __init__(self, lattice, flow):
        #wird wahrscheinlich nicht gehen wegen fft
        #init sollte gehen
        super(EnergySpectrum, self).__init__(lattice, flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.grid[0].shape
        #das sollte gleich sein
        frequencies = [self.lattice.convert_to_tensor(np.fft.fftfreq(dim, d=1 / dim)) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies))
        wavenorms = torch.norm(wavenumbers, dim=0)

        if self.lattice.D == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
                (wavenorms[..., None] > self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) - 0.5) &
                (wavenorms[..., None] <= self.wavenumbers.to(dtype=lattice.dtype, device=lattice.device) + 0.5)
        )

    def __call__(self, f):
        u = self.lattice.u(f)
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        ekin = self._ekin_spectrum(u)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.lattice.dtype)
        ek = ek.sum(torch.arange(self.lattice.D).tolist())
        return ek

    def _ekin_spectrum(self, u):
        """distinguish between different torch versions"""
        torch_ge_18 = (version.parse(torch.__version__) >= version.parse("1.8.0"))
        if torch_ge_18:
            return self._ekin_spectrum_torch_ge_18(u)
        else:
            return self._ekin_spectrum_torch_lt_18(u)

    def _ekin_spectrum_torch_lt_18(self, u):
        zeros = torch.zeros(self.dimensions, dtype=self.lattice.dtype, device=self.lattice.device)[..., None]
        uh = (torch.stack([
            torch.fft(torch.cat((u[i][..., None], zeros), self.lattice.D),
                      signal_ndim=self.lattice.D) for i in range(self.lattice.D)]) / self.norm)
        ekin = torch.sum(0.5 * (uh[..., 0] ** 2 + uh[..., 1] ** 2), dim=0)
        return ekin

    def _ekin_spectrum_torch_ge_18(self, u):
        ##fft über ganze domaine wird nicht gehen auser wir machen eine kollective operation auf einem Knoten
        uh = (torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.lattice.D))) for i in range(self.lattice.D)
        ]) / self.norm)
        ekin = torch.sum(0.5 * (uh.imag ** 2 + uh.real ** 2), dim=0)
        return ekin


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundaries).
    """

    def __init__(self, lattice, flow, no_mass_mask=None, mpiObj=None):
        super(Mass, self).__init__(lattice, flow)
        self.lattice=lattice
        self.flow=flow
        self.mask = no_mass_mask
        self.mpiObj=mpiObj
        if(self.mpiObj is not None):
            if(mpiObj.mpi==1):
                #we have a distributed array!
                global dist
                import torch.distributed as dist 
                #set up warnings for that case       
                if(lattice.dtype==torch.float32):
                    print("Warning: Because float addition is not associativ results may differ")
                self.calling=self.mpiCall
            
            else:
                self.calling=self.nonMPIcall

        else:
            self.mpiObj=lettuce.mpiClass.mpiObject(0)
            self.calling=self.nonMPIcall
        
    def __call__(self, f):
        return self.calling(f)

    def nonMPIcall(self,f):
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()
        return mass

    def mpiCall(self,f):
        #calculating the same way as NonMPI
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()

        #sum reduction to rank 0
        dist.reduce(mass,0,dist.ReduceOp.SUM)
        return mass