import pickle

from torch._C import CompilationUnit

class mpiObject:
    """This is the MPI-Object it will handle all MPI-realated tasks"""

    def __init__(self, activateMPI, sizeList=None, gpuList=None, setParts=0,distributefromRank0=0, initOnCPU=0, gridRefinment=0):
        """ActivateMPI - Should the computeation use mpi
           sizeList - How much should a rank cover eg. [[0,40],[5,10]] With this example Rank 0 will cover 40 slizes rank 2 will take up the slack, rank 5 only covers 10
           gpuList - Numbers of GPUs per PC
           setParts - use the Userdefined sizeList
           distributefromRank0 - should the inital Flow from Rank 0 be distributed
           initOnCPU - the inital calculation will be made on the CPU
           gridRefinment - should the grid be refint? size and stepcount are doubled """
        self.mpi=activateMPI
        self.initOnCPU=initOnCPU
        self.device=0
        self.distributefromRank0=distributefromRank0
        self.nodeList=None
        self.gpuList=None
        self.next=0
        self.prev=0
        self.rank=0
        self.size=1

        self.gridRefinment=gridRefinment
        if(self.mpi):
            
            self.nodeList=sizeList
            self.gpuList=gpuList
            self.setParts=setParts

    
        self.index=[]

        self.name=""
        self.activeNodes=[]




class running(object):
    def __init__(self,methodeToRun, pytortchDevice, mpiObjectInput=None):
        mpiObj=mpiObjectInput
        if(mpiObjectInput==None):
            mpiObj=mpiObject(0)

        self.makeImports(mpiObj.mpi)
        
        finalMPIObject=self.distribute(mpiObj,pytortchDevice)
        methodeToRun(finalMPIObject.device, finalMPIObject)

    def makeImports(self,mpi):
        if(mpi):
            #run distributed
            global os
            import os

            global dist
            import torch.distributed as dist

            global torch
            import torch
    
    def _init_processes_mpi(self, device , mpiObj ):
        """ Initialize the distributed environment. """
        dist.init_process_group("mpi")
        size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])


        os_info = os.uname()
        pcname=os_info[1]
        mpiObj.name=pcname
        print(f"Process {rank} of {size} starting up on {pcname}!")
        
        mpiObj.size=size
        mpiObj.rank=rank
        if(size!=1):
            nodeList=self.computationList(rank, size,pcname)
            mpiObj.activeNodes=nodeList
            if device.type == "cuda":
                #wer ist mit mir auf der gleichen node?
                assert(mpiObj.gpuList is not None, "No gpu list given please insert gpu list wen creating the MPI-object")

                neighbours=self.myNeighbours(nodeList,pcname)
                #verteile die Cuda resourcen
                numGPUs= self.getNumberOfGpus(mpiObj.gpuList, pcname)
                myDevice=self.distrubuteCuda(neighbours,numGPUs,device,rank)
                mpiObj.device=myDevice

            else:
                mpiObj.device=device

            #set communication    
            self.communicationList(mpiObj,rank, size)

        else:
            mpiObj=mpiObject(0)
            mpiObj.device=device

        
        print(f"Process {rank} using device {mpiObj.device}")
        
        return mpiObj

    def distribute(self,mpiObj, device):

        if(mpiObj.mpi):
            mpiObj=self._init_processes_mpi(device,mpiObj)
        else:
            mpiObj.device=device

        return mpiObj
    
    def communicationList(self,mpiObj,rank,size):
        #set next and previous node
        nextone=-1
        prevone=-1
        #set next one
        if(rank!=size-1):
            #if i am not the last one the next one is myrank+1
            nextone=rank+1
        else:
            #if i am the last one my next one is rank 0 because myrank+1==size doesnt exist
            nextone=0

        #set prevone
        if(rank!=0):
            #if i am not the first one the prev one is myrank-1
            prevone=rank-1
        else:
            #if i am the first one my prev one is rank size-1
            prevone=size-1
    
        mpiObj.prev=prevone
        mpiObj.next=nextone

    def getNumberOfGpus(self,gpuList, myname):
        for entry in gpuList:
            nodeName=entry[1]
            if(nodeName==myname):
                return entry[0]
        return -1

    def distrubuteCuda(self,neighbours,numGPUs,device,rank):
        #get my node
        #in my node determent rank
        myIndex=neighbours.index(rank)
        #get num of cuda
        #set  cuda device
        print(rank,myIndex,numGPUs)
        if(myIndex<numGPUs):
            #torch.cuda.set_device(myIndex)
            device = torch.device(f"cuda:{myIndex}")

        else:
            #or set cpu
            device = torch.device("cpu")

        return device
    
    def computationList(self,rank,size, pcname):
        """get all Node pcnames"""
        #get Node name
        
        #info is rank + pc name
        info=str(rank)+","+pcname
        #convert it into ascii
        asciiList=[]
        for letter in info:
            asciiList.append(ord(letter))
        #get length of the asciiList
        maxlength=torch.Tensor([len(asciiList)])
        llength=len(asciiList)

        #get from all nodes the longest version
        dist.all_reduce(maxlength, op=dist.ReduceOp.MAX)

        #if not long eneught fill it up with ","
        maxlength=int(maxlength.item())
        while llength!=maxlength:
            asciiList.append(ord(","))
            llength=len(asciiList)
        #create tensor of our asciiList
        tensor = torch.Tensor(asciiList)

        #create input buffer
        allnodes = [torch.ones((maxlength)) for _ in range(size)]
        #get all information from other nodes
        dist.all_gather(allnodes,tensor)


        #convert the list tensor into list,list
        nodelist=[]
        for i in range(size):
            nodelist.append(allnodes[i].tolist())

        #retranslate the ascii numbers to chars and split rank and pc node name and if pc node name has any "," remove them

        nodeString = []
        for i in range(size):
            string=""
            ranknumber=""
            num=True
            for char in nodelist[i]:
                if(num):
                    iscomma=chr(int(char))
                    if(iscomma!=","):
                        ranknumber=ranknumber+chr(int(char))
                    else:
                        num=False
                else:
                    
                    iscomma=chr(int(char))
                    if(iscomma!=","):
                        string = string + chr(int(char))
                    else:
                        pass
            #convert ranknumber to number
            ranknumber=int(ranknumber)
            #create tuple to insert
            insert=[ranknumber,string]
            nodeString.append(insert)

        return nodeString

    def myNeighbours(self,nodeList,pcname):

        my_neibours=[]

        for entry in nodeList:
            other_rank=entry[0]
            other_pcname=entry[1]
            if(other_pcname==pcname):
                my_neibours.append(other_rank)

        return my_neibours
