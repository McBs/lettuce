import numpy as np
from copy import deepcopy
from lettuce import LettuceException, mpiClass
import torch.distributed as dist
import torch

__all__ = ["RegularGrid"]

class RegularGrid(object):

    def __init__(self, resolution, char_length_lu, char_length_pu, endpoint=False, mpiObject=None,lattice=None):
        """
        class to construct a regular lattice grid for the simulation
        using the rank and size arguments this can be used to split the simulation domain across several processes

        Input parameters:
        resolution: list of up to three values for the resolution in x, y, (z)
        char_length_lu/pu: characteristic length of the flow in lu or pu respectively
        endpoint: True if the end of the domain shall be included in the grid; e.g. if 0 to 2pi is [0, 2pi] instead of [0, 2pi)
        rank: rank of the process constructing the grid
        size: total number of processes

        Usable parameters:
        self: returns grid as list of two / three elements, one coordinate each [x, y, z]
        shape: returns shape of one of the grid-array without having to construct it first

        functions:
        --select:
        Inputs:
        tensor (or numpy array) of the size of the grid (if the input is 4D the last three will be assumed to be the grid coordinates)
        rank (optional)

        Output:
        Tensor with part of the input tensor that belongs to process "rank" (calling process if rank is empty / None)

        --reassemble:
        Inputs:
        tensor: tensor to be reassembled (pytorch tensor that is present on all processes)

        Output:
        on process with rank 0: the whole tensor (of the full domain)
        on all other processes: 1
        """
        self.lattice=lattice
        self.resolution = resolution
        self.char_length_lu = char_length_lu
        self.char_length_pu = char_length_pu
        self.endpoint = endpoint
        if(mpiObject!=None):
            #if mpi is not used use these values to simulate the calculations for only one pc
            if(mpiObject.mpi==0):
                self.rank = 0
                self.size = 1
                self.index = slice(int(np.floor(self.resolution[0] * self.rank / self.size)),
                           int(np.floor(self.resolution[0] * (self.rank + 1) / self.size)))
            else:        
                self.rank = mpiObject.rank
                self.size = mpiObject.size
                #if set Parts is selected calculate new Slices
                if(mpiObject.setParts==1 ):
                    
                    computeList=[]

                    for i in range(self.size):
                        indexI=slice(int(np.floor(resolution[0] * i / self.size)),
                                        int(np.floor(resolution[0] * (i+1) / self.size)))
                        computeList.append(indexI)
                    #instructions on how to resize the arrays
                    liste=mpiObject.nodeList
                    #list of all nodes in rank oder
                    others=mpiObject.activeNodes
                    
                    #go over the rank list and look up the reverence Table
                    #if the entry matches update the slice
                    for anode in others:
                        anodeID=anode[0]
                        for j in range(len(liste)):
                            listednode=liste[j]                            
                            lrank=listednode[0]
                            if(anodeID==lrank):
                                newSize=listednode[1]
                                if(anodeID!=self.size-1):
                                    #set values for the node
                                    entry=computeList[anodeID]

                                    start=entry.start
                                    end=start+newSize
                                    newslice=slice(start,end)
                                    computeList[anodeID]=newslice

                                    #edit the next node to take up the left out space
                                    entry=computeList[anodeID+1]
                                    start=end
                                    end=entry.stop
                                    newslice=slice(start,end)
                                    computeList[anodeID+1]=newslice

                                else:
                                    #set values for the node
                                    entry=computeList[anodeID]
                                    end=entry.stop
                                    start=end-newSize
                                    newslice=slice(start,end)
                                    computeList[anodeID]=newslice

                                    #edit the prev node to take up the left out space
                                    entry=computeList[anodeID-1]
                                    end=start
                                    start=entry.stop
                                    
                                    newslice=slice(start,end)
                                    computeList[anodeID+1]=newslice

                             
                    #everyone has the list now and saves it
                    myself=computeList[self.rank]
                    self.index=myself
                    self.computeList=computeList

                else:
                    #no correction
                    self.index = slice(int(np.floor(self.resolution[0] * self.rank / self.size)),
                           int(np.floor(self.resolution[0] * (self.rank + 1) / self.size)))
                    print("INIT",self.index)
                    computeList=[]

                    for i in range(self.size):
                        indexI=slice(int(np.floor(resolution[0] * i / self.size)),
                                        int(np.floor(resolution[0] * (i+1) / self.size)))
                        computeList.append(indexI)

                    self.computeList=computeList


            
        else:
            mpiObject=mpiClass.mpiObject(0)        
            self.rank = 0
            self.size = 1
            self.index = slice(int(np.floor(self.resolution[0] * self.rank / self.size)),
                           int(np.floor(self.resolution[0] * (self.rank + 1) / self.size)))
        

        mpiObject.index=self.index
        self.shape = deepcopy(self.__call__()[0].shape)
        self.shape_pu = [res * self.char_length_pu / self.char_length_lu for res in self.resolution]
        temp=deepcopy(self.resolution)
        temp.insert(0,9)
        self.global_shape = torch.Size(temp)

    def __call__(self):
        x = np.linspace(0 + self.index.start * self.char_length_pu / self.char_length_lu,
                        self.index.stop * self.char_length_pu / self.char_length_lu,
                        num=self.index.stop - self.index.start, endpoint=self.endpoint)
        y = np.linspace(0, self.resolution[1] * self.char_length_pu / self.char_length_lu, num=self.resolution[1], endpoint=self.endpoint)
        if len(self.resolution) == 3:
            z = np.linspace(0, self.resolution[2] * self.char_length_pu / self.char_length_lu, num=self.resolution[2], endpoint=self.endpoint)
            return np.meshgrid(x, y, z, indexing='ij')
        else:
            return np.meshgrid(x, y, indexing='ij')

    def global_grid(self):
        x = np.linspace(0, self.resolution[0] * self.char_length_pu / self.char_length_lu, num=self.resolution[0], endpoint=self.endpoint)
        y = np.linspace(0, self.resolution[1] * self.char_length_pu / self.char_length_lu, num=self.resolution[1], endpoint=self.endpoint)
        if len(self.resolution) == 3:
            z = np.linspace(0, self.resolution[2] * self.char_length_pu / self.char_length_lu, num=self.resolution[2], endpoint=self.endpoint)
            return np.meshgrid(x, y, z, indexing='ij')
        else:
            return np.meshgrid(x, y, indexing='ij')

    def select(self, tensor, rank=None):
        """reduce tensor (or numpy-array) to the part associated with process "rank" (calling process if rank is None / empty)"""
        if rank is not None:
            assert rank < self.size, LettuceException(f"Calling RegularGrid.select with "
                                                      f"rank ({rank}) >= size ({self.size}), expected rank < size.")
            assert rank >= 0 and (type(rank) is int or type(rank) is float), \
                LettuceException("Calling RegularGrid.select with wrong "
                                 f"rank = {rank} (type: {type(rank)}), expected rank to be a positive int or float.")
            index = slice(int(np.floor(self.resolution[0] * rank / self.size)),
                           int(np.floor(self.resolution[0] * (rank + 1) / self.size)))
        else:
            index = self.index

        if len(tensor.shape) > len(self.resolution):
            return tensor[:, index, ...]
        else:
            return tensor[index, ...]

    def cconvert_coordinate_global_to_local(self, coordinates):
        """converts global coordinates into local coordinates in domain of this process"""
        assert ((coordinates[0] > self.index.start) and (coordinates[0] < self.index.stop)), \
            Exception(
                f"The domain of the process with rank {self.rank} does not contain {coordinates})!")
        coordinates[0] = coordinates[0] - self.index.start
        return coordinates

    def convert_coordinate_local_to_global(self, coordinates):
        """converts local coordinates into global coordinates"""
        coordinates[0] = coordinates[0] + self.index.start
        return coordinates

    def reassemble(self, tensor):
        """recombines tensor that is spread to all processes in process 0
        (should just return tensor if only one process exists)"""
        if self.rank == 0:
            assembly = tensor
            for i in range(1, self.size):
                if len(tensor.shape) > len(self.shape):
                    input = self.select(torch.zeros([tensor.shape[0]] + self.resolution,
                                                    device=tensor.device, dtype=tensor.dtype), rank=i).contiguous()
                    dist.recv(tensor=input, src=i)
                    assembly = torch.cat((assembly, input), dim=1)
                else:
                    input = self.select(torch.zeros(self.resolution,
                                                    device=tensor.device, dtype=tensor.dtype), rank=i).contiguous()
                    dist.recv(tensor=input, src=i)
                    assembly = torch.cat((assembly, input), dim=0)
            return assembly
        else:
            output = tensor.contiguous()
            dist.send(tensor=output, dst=0)
            return 1

    def distributeToList(self,tensor,Q=-1):
        """Distributes the relevant part of a tensor to the specific rank """
        if(self.rank==0):
            for i in range(1,self.size):
                selectindex=self.computeList[i]
                sending=tensor[:,selectindex.start:selectindex.stop,...]
                trans=sending.detach().clone().cpu().contiguous()
                dist.send(tensor=trans,dst=i)
            selectindex=self.computeList[0]
            return tensor[:,selectindex.start:selectindex.stop,...]
        else:
            res=[]
            if(Q==-1):
                res.append(self.lattice.Q)
            res.append(self.resolution[0])
            res.append(self.resolution[1])
            
            if(self.lattice.Q==3):
                res.append(self.resolution[2])
            
            res[1]=self.index.stop-self.index.start
            local=torch.zeros(res)
            dist.recv(tensor=local, src=0)
            return local
