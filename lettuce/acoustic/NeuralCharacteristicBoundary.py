import torch
import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import warnings
from typing import Union, List, Optional
from lettuce import UnitConversion, D2Q9, ExtFlow
from matplotlib.patches import Rectangle
from utility import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from itertools import product
from plot import *
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import torch.optim as optim



def run(context, config, K, dataset, dataset_nr, t_pu):
    # with torch.no_grad():
    flow = Acoustic(context, [config["nx"]+config["extension"], config["ny"]],
                    reynolds_number=config["Re"],
                    mach_number=config["Ma"],
                    velocity_init=1,
                    K=K,
                    distanceFromRight=200+config["extension"])
    collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
    if config["reporter"]:
        TotalPressureReporter = TotalPressure(context=context, interval=int(flow.units.convert_time_to_lu(0.05)), slices=slices_2)
        simulation.reporter.append(TotalPressureReporter)
        # simulation.reporter.append(lt.VTKReporter(context, flow, interval=int(flow.units.convert_time_to_lu(0.05)), filename_base="vtkoutput/out"))
    if config["save_dataset"]:
        print(f"Saving dataset for Mach {config["Ma"]:03.2f} every {config["save_iteration"]:2.2f} seconds")
        # hdf5_reporter = HDF5Reporter(
        #              flow=flow,
        #              slices=None,
        #              context=context,
        #              interval= config["save_iteration"],
        #              t_pu = t_pu,
        #              filebase=f"./dataset_mach-{config["Ma"]:03.2f}_interv-{config["save_iteration"]:03.2f}",
        #              trainingsdomain=slices,
        #              )
        tensor_reporter = TensorReporter(
            flow=flow,
            interval = config["save_iteration"],
            t_pu = t_pu,
            filebase=f"./dataset_mach-{config["Ma"]:03.2f}_interv-{config["save_iteration"]:03.2f}",
            trainingsdomain=slices
        )
        # simulation.reporter.append(hdf5_reporter)
        simulation.reporter.append(tensor_reporter)
    if config["load_dataset"] and dataset_nr is not None and callable(dataset_train):
        simulation.flow.f = dataset(dataset_nr)[:,:config["nx"]+config["extension"],:config["ny"]]
    with torch.set_grad_enabled(config["train"]):
        # simulation(num_steps=1)
        t_lu = round(flow.units.convert_time_to_lu(t_pu))
        print(f"t_lu = {t_lu}")
        simulation(num_steps=t_lu)
        # simulation.boundaries[1].K = 0.4
        # simulation(num_steps=int(flow.units.convert_time_to_lu(1)))
    reporter = simulation.reporter[0] if config["reporter"] else None
    return flow, reporter

class NeuralTuning(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device='cuda', nodes=20, index=None):
        """Initialize a neural network boundary model."""
        super(NeuralTuning, self).__init__()
        self.moments = D2Q9Dellar(lt.D2Q9(), lt.Context(device="cuda", dtype=torch.float64, use_native=False))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, nodes, bias=True),
            torch.nn.Linear(nodes, 1, bias=True),
            torch.nn.Sigmoid(),
        ).to(dtype=dtype, device=device)
        self.index = index
        self.max = 0
        self.min = 1

        print("Initialized NeuralTuning")

    def forward(self, f):
        """Forward pass through the network with residual connection."""
        local_moments = self.moments.transform(f.unsqueeze(1))
        K = self.net(local_moments[:,0,:].transpose(0,1))
        self.max = K.max() if K.max() > self.max else self.max
        self.min = K.min() if K.min() < self.min else self.min
        return K

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nx", type=int, default=250)
    parser.add_argument("--ny", type=int, default=350)
    parser.add_argument("--extension", type=int, default=0)
    parser.add_argument("--Re", type=int, default=750, help="")
    parser.add_argument("--Ma", type=float, default=0.3, help="")
    parser.add_argument("--t_pu", type=float, default=6)
    parser.add_argument("--load_dataset", action="store_true", default=True)
    parser.add_argument("--load_dataset_idx", type=int, default=None)
    parser.add_argument("--save_dataset", action="store_true", default=False)
    parser.add_argument("--save_iteration", type=float, default=0.25)
    parser.add_argument("--K_neural", action="store_true", default=True)
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--load_model", action="store_true", default=True)
    parser.add_argument("--model_name_saved", type=str, default="model_trained_v1.pt")
    parser.add_argument("--model_name_loaded", type=str, default="model_trained.pt")
    parser.add_argument("--reporter", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_step", type=int, default=10)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--train_mach_numbers", type = float, nargs = "+", default = [0.3])
    parser.add_argument("--train_t_pu_intervals", type=int,  nargs="+", default=[12, 24])
    parser.add_argument("--expand_intervals", action="store_true", default=False)
    parser.add_argument("--slices", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--loss_plot_name", type=str, default="model_trained.pt")
    args, unknown = parser.parse_known_args()
    args = vars(args)
    [print(arg, args[arg]) for arg in args]
    shift = 7
    torch.manual_seed(0)

    K_tuned = NeuralTuning() if args["K_neural"] else 0.4
    if args["load_model"] and args["K_neural"]:
        K_tuned = torch.load(args["model_name_loaded"], weights_only=False)
        K_tuned.eval()
        print("Model loaded")
    context = lt.Context(torch.device("cuda:0"), use_native=False, dtype=torch.float64)
    slices = [slice(args["nx"] - 200, args["nx"]-1), slice(args["ny"] // 2 - 100, args["ny"] // 2 + 100)]
    slices_2 = [slice(args["nx"] - 200, args["nx"]-150), slice(args["ny"] // 2 - 100, args["ny"] // 2 + 100)]
    slices_all = [slice(None, None), slice(None, None)]
    # slices = [slice(None, None), slice(None, None)]

    machNumbers = args["train_mach_numbers"]
    intervals = args["train_t_pu_intervals"]
    if args["expand_intervals"]:
        intervals = [x for x in intervals for _ in range(x)]

    pairs = list(product(intervals, machNumbers)) if args["train"] else [(1,args["Ma"])]
    print("Configurations: ", len(pairs))
    if callable(K_tuned) and args["train"]:
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(K_tuned.parameters(), lr=args["lr"])
        if args["scheduler"]:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args["scheduler_step"], gamma=args["scheduler_gamma"])
        epoch_training_loss = []
        scaler = GradScaler()
        optimizer.zero_grad()

    for _ in range(args["epochs"] if args["train"] else 1):
        print(f"Epoch: {_}" if args["train"] else "Running ...")
        running_loss = 0.0
        for i, (idx, ma) in enumerate(pairs):

            dataset_name = f"./dataset_mach-{ma:03.2f}_interv-{args["save_iteration"]:03.2f}.h5"
            if args["load_dataset"] or args["train"]:
                # dataset_train = LettuceDataset(context=context, filebase=dataset_name, target=False)
                dataset_train = TensorDataset(
                                 file_pattern= f"./dataset_mach-{ma:03.2f}_interv-{args["save_iteration"]:03.2f}_*",
                                 transform = None
                )


                slices_ref = dataset_train.get_trainingsdomain()
                slices_ref = [slice(slices_ref[0], slices_ref[1]), slice(slices_ref[2], slices_ref[3])]
            else:
                dataset_train = None
            if args["train"]: optimizer.zero_grad()
            t_pu = idx * args["save_iteration"] if args["train"] else args["t_pu"]
            print(i, ma, t_pu, idx)
            with autocast(context.device.type):
                flow, reporter = run(context=context,
                                     config=args,
                                     K=K_tuned,
                                     dataset = dataset_train,
                                     dataset_nr = args["load_dataset_idx"],
                                     t_pu = t_pu
                                     )
            if callable(K_tuned) and args["train"]:
                offset = 0 if args["load_dataset_idx"] is None else args["load_dataset_idx"]
                reference = dataset_train(idx+offset)[0].to(device="cuda")[:,slices_ref[0],slices_ref[1]]
                rho_ref = flow.rho(reference)
                rho_train = flow.rho()[:,slices[0],slices[1]]
                # loss = criterion(flow.f[:,slices[0],slices[1]], reference)
                k = K_tuned(flow.f[:,slices[0].stop-1,:])
                loss = criterion(rho_ref, rho_train) #+ criterion(k, torch.zeros_like(k))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                optimizer.zero_grad()
                if args["verbose"]: print("running_loss:", running_loss)


        if args["train"]:
            # plot_velocity_density(flow.f, flow=flow, config=args, slices=slices, rectangle=False)
            if args["scheduler"]:
                scheduler.step()
            epoch_training_loss.append(running_loss)
            if args["verbose"]: print(epoch_training_loss)

    if args["train"]: torch.save(K_tuned, args["model_name_saved"])



    slices_plot = slices if args["slices"] else slices_all
    rectangle = False if args["slices"] else True




    plot_velocity_density(flow.f, flow=flow, config=args, slices=slices, rectangle=False)
    # plotRho(flow.f, flow=flow, config=args, slices=slices_plot, rectangle=rectangle)

    if args["load_dataset"]:
        if args["train"] is False:
            ref = dataset_train(int(args["t_pu"]/args["save_iteration"]))[0].to(device="cuda")
        else:
            ref = reference
            rectangle = False
            plot_velocity_density(ref, flow=flow, config=args, slices=slices_all, rectangle=False)
        slices_ref = dataset_train.get_trainingsdomain()
        slices_ref = [slice(slices_ref[0], slices_ref[1]), slice(slices_ref[2], slices_ref[3])]

        # plot_velocity_density(ref, flow=flow, config=args, slices=slices_ref, rectangle=False)
        # plotRho(ref, flow=flow, config=args, slices=slices_plot , rectangle=rectangle)


    if args["K_neural"]:
        print(K_tuned(flow.f[:,slices[0].stop,:]))
    if args["train"]:
        plot = PlotNeuralNetwork(base="./", show=True, style="./ecostyle.mplstyle")
        plot.loss_function(np.array(epoch_training_loss)/epoch_training_loss[0], name=args["loss_plot_name"])
        print("K tuned max: ", K_tuned.max)
        print("K tuned min: ", K_tuned.min)

    if reporter is not None:
        out = torch.tensor(reporter.out_total).cpu().detach()
        t = torch.tensor(reporter.t).cpu().detach()
        plt.plot(t,out)
        plt.ylim(1-1e-5,1+1.3e-5)
        plt.show()
    print((ref-flow.f[:,slices[0],slices[1]]).sum())