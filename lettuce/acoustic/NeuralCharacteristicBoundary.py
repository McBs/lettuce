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



def run(context, config, K, dataset, dataset_nr, t_lu):
    # with torch.no_grad():
    flow = Acoustic(context, [config["nx"]+config["extension"], config["ny"]],
                    reynolds_number=config["Re"],
                    mach_number=config["Ma"],
                    velocity_init=1,
                    K=K,
                    xc=config["xc"],
                    distanceFromRight=200+config["extension"])
    collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
    if config["reporter"]:
        TotalPressureReporter = TotalPressure(context=context, interval=int(flow.units.convert_time_to_lu(0.05)), slices=slices_2)
        simulation.reporter.append(TotalPressureReporter)
        # simulation.reporter.append(lt.VTKReporter(context, flow, interval=int(flow.units.convert_time_to_lu(0.05)), filename_base="vtkoutput/out"))
    if config["save_dataset"]:
        print(f"Saving dataset for Mach {config["Ma"]:03.2f} every {config["save_iteration"]:2.2f} seconds")
        tensor_reporter = TensorReporter(
            flow=flow,
            interval = config["save_iteration"],
            t_lu = t_lu,
            filebase=f"./{config["output_directory"]}/dataset_mach-{config["Ma"]:03.2f}_interv-{config["save_iteration"]:03.2f}",
            trainingsdomain=slices_training,
            start_idx=config["save_start_idx"],
        )
        # simulation.reporter.append(hdf5_reporter)
        simulation.reporter.append(tensor_reporter)
    if config["load_dataset"] and dataset_nr is not None and callable(dataset_train):
        simulation.flow.f = dataset.get_f(dataset_nr)
    if config["load_dataset"] and dataset_nr is None and callable(dataset_train) and config["load_dataset_path"] is not None:
        simulation.flow.f = dataset._load_and_process_f(config["load_dataset_path"])
    simulation.boundaries[1].rho_t1 = flow.rho()[0, -1, :]
    simulation.boundaries[1].u_t1 = flow.u()[0, -1, :]
    simulation.boundaries[1].v_t1 = flow.u()[1, -1, :]
    if config["train"]:
        f_pre = dataset.get_f(dataset_nr - 1)
        simulation.boundaries[1].rho_dt_old = flow.rho(f_pre)[0, -1, :] - simulation.boundaries[1].rho_t1
        simulation.boundaries[1].u_dt_old = flow.u(f_pre)[0, -1, :] - simulation.boundaries[1].u_t1
        simulation.boundaries[1].v_dt_old = flow.u(f_pre)[1, -1, :] - simulation.boundaries[1].v_t1
    with torch.set_grad_enabled(config["train"]):
        # simulation(num_steps=1)
        # print(f"t_lu = {t_lu}")
        simulation(num_steps=int(t_lu))
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
            torch.nn.Linear(nodes, nodes, bias=True),
            torch.nn.BatchNorm1d(nodes),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(nodes, 2, bias=True),
        ).to(dtype=dtype, device=device)
        self.index = index
        self.K0max = 0
        self.K0min = 1
        self.K1max = 0
        self.K1min = 1

        print("Initialized NeuralTuning")

    def forward(self, f, p_dx, u_dx, v_dx, p_dy, u_dy, v_dy, p_dt, u_dt, v_dt, velocity_init=0):
        """Forward pass through the network with residual connection."""
        local_moments = self.moments.transform(f.unsqueeze(1))
        # K = self.net(local_moments[:,0,:].transpose(0,1))
        rho = local_moments[0,:,:].transpose(0,1)
        u = torch.abs(local_moments[1, :, :] - velocity_init[0]).transpose(0,1)
        v = torch.abs(local_moments[2, :, :]).transpose(0,1)
        K = self.net(
            torch.cat([
                p_dx.unsqueeze(1),
                u_dx.unsqueeze(1),
                v_dx.unsqueeze(1),
                p_dy.unsqueeze(1),
                u_dy.unsqueeze(1),
                v_dy.unsqueeze(1),
                p_dt.unsqueeze(1),
                u_dt.unsqueeze(1),
                v_dt.unsqueeze(1)], dim=1)
        )
        # K = self.net(
        #     torch.cat([
        #         local_moments[3:,0,:].transpose(0,1),
        #         rho_dt.unsqueeze(1),
        #         u_dt.unsqueeze(1),
        #         v_dt.unsqueeze(1)], dim=1)
        # )
        K = torch.nn.Sigmoid()(K)
        self.K0max = K[0].max() if K[0].max() > self.K0max else self.K0max
        self.K0min = K[0].min() if K[0].min() < self.K0min else self.K0min
        self.K1max = K[1].max() if K[1].max() > self.K1max else self.K1max
        self.K1min = K[1].min() if K[1].min() < self.K1min else self.K1min
        K[:,1] = K[:,1]*4
        return K

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nx", type=int, default=300)
    parser.add_argument("--ny", type=int, default=500)
    parser.add_argument("--extension", type=int, default=0)
    parser.add_argument("--Re", type=int, default=750, help="")
    parser.add_argument("--Ma", type=float, default=0.3, help="")
    parser.add_argument("--xc", type=int, default=150)
    parser.add_argument("--t_lu", type=int, default=300)
    parser.add_argument("--load_dataset", action="store_true", default=False)
    parser.add_argument("--load_dataset_idx", type=int, default=0)
    # parser.add_argument("--load_dataset_path", type=str, default="datasets/dataset_mach-0.30_interv-55.00_000055.pt")
    # parser.add_argument("--load_dataset_path", type=str, default="datasets/dataset_mach-0.30_interv-210.00_000210.pt")
    parser.add_argument("--load_dataset_path", type=str, default=None)
    parser.add_argument("--show_dataset_idx", type=int, default=None)
    parser.add_argument("--save_dataset", action="store_true", default=False)
    parser.add_argument("--save_iteration", type=int, default=5)
    parser.add_argument("--save_start_idx", type=float, default=0)
    parser.add_argument("--K_neural", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--model_name_saved", type=str, default="model_trained_v6.pt")
    parser.add_argument("--model_name_loaded", type=str, default="model_training_v5_1.pt")
    parser.add_argument("--output_directory", type=str, default="datasets")
    parser.add_argument("--reporter", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--scheduler_step", type=int, default=130)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--training_iteration", type=int, nargs="+", default = [5, 10, 15])
    parser.add_argument("--train_mach_numbers", type = float, nargs = "+", default = [0.3])
    parser.add_argument("--train_t_lu_intervals", type=int, nargs="+", default=[1, 100, 1])
    parser.add_argument("--train_t_lu_intervals_2", type=int, nargs="+", default=None)
    parser.add_argument("--expand_intervals", action="store_true", default=False)
    parser.add_argument("--slices", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--loss_plot_name", type=str, default="model_trained.pt")
    args, unknown = parser.parse_known_args()
    args = vars(args)
    [print(arg, args[arg]) for arg in args]
    shift = 7
    torch.manual_seed(0)
    np.random.seed(0)

    K_tuned = NeuralTuning() if args["K_neural"] else [1, 0]
    if args["load_model"] and args["K_neural"]:
        K_tuned = torch.load(args["model_name_loaded"], weights_only=False)
        K_tuned.eval()
        print("Model loaded")
    if args["train"] and callable(K_tuned):
        K_tuned.train()
    context = lt.Context(torch.device("cuda:0"), use_native=False, dtype=torch.float64)
    slices_training = [slice(args["nx"] - 200, args["nx"]-1), slice(args["ny"] // 2 - 100, args["ny"] // 2 + 100)]
    slices_domain = [slice(0, args["nx"]+args["extension"]), slice(0, args["ny"])]
    slices_2 = [slice(args["nx"] - 200, args["nx"]-150), slice(args["ny"] // 2 - 100, args["ny"] // 2 + 100)]
    # slices_all = [slice(None, None), slice(None, None)]
    # slices = [slice(None, None), slice(None, None)]

    machNumbers = args["train_mach_numbers"]
    intervals = np.arange(*args["train_t_lu_intervals"])
    intervals_2 = np.arange(*args["train_t_lu_intervals_2"]) if args["train_t_lu_intervals_2"] is not None else []
    intervals = np.concatenate((intervals, intervals_2))
    training_iterations = args["training_iteration"]
    if args["shuffle"]: intervals = np.random.permutation(intervals)
    load_dataset_idx = args["load_dataset_idx"] if args["load_dataset"] else 0
    pairs = list(product(intervals, machNumbers, training_iterations)) if args["train"] else [(load_dataset_idx, args["Ma"], 0)]
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
        for i, (idx, ma, training_iteration) in enumerate(pairs):

            dataset_name = f"./{args["output_directory"]}/dataset_mach-{ma:03.2f}_interv-{args["save_iteration"]:03.2f}_*"
            if args["load_dataset"] and args["load_dataset_path"] is not None:
                dataset_name = (args["load_dataset_path"])
            if args["K_neural"]:
                K_tuned.max = 0
                K_tuned.min = 1
            if args["load_dataset"] or args["train"]:
                dataset_train = TensorDataset(
                                 file_pattern= dataset_name,
                                 transform = None,
                                 slices_domain = slices_domain,
                                 verbose=args["verbose"],
                                 device="cuda",
                                )
            else:
                dataset_train = None
            if args["train"]: optimizer.zero_grad()
            t_lu = training_iteration if args["train"] else args["t_lu"]
            # idx=0
            print(f"pair idx {i}, mach: {ma}, t_lu: {t_lu}, loaded dataset idx: {idx}, loaded reference idx: {int(idx+t_lu/args["save_iteration"])}",)
            if args["train"] and int(idx+t_lu/args["save_iteration"])>600:
                print("continue")
                continue
            with autocast(context.device.type):
                flow, reporter = run(context=context,
                                     config=args,
                                     K=K_tuned,
                                     dataset = dataset_train,
                                     dataset_nr = int(idx),
                                     t_lu = t_lu,
                                     )
            if callable(K_tuned) and args["train"]:
                offset = 0 if args["load_dataset_idx"] is None else args["load_dataset_idx"]
                reference = dataset_train.get_f(int(idx+t_lu/args["save_iteration"]), True)
                rho_ref = flow.rho(reference)[:,-training_iteration:,50:150]
                rho_train = flow.rho()[:,*slices_training][:,-training_iteration:,50:150]
                u_ref = flow.u(reference)[:,-training_iteration:,50:150]
                u_train = flow.u()[:,*slices_training][:,-training_iteration:,50:150]
                # loss = criterion(flow.f[:,slices[0],slices[1]], reference)
                # k = K_tuned(flow.f[:,slices[0].stop-1,:],3*[torch.zeros_like(flow.f[0,-1,:])])
                loss = criterion(rho_ref, rho_train)# + criterion(u_ref, u_train) #+ criterion(k, torch.zeros_like(k))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # loss.backward()
                # optimizer.step()
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



    # slices_plot = slices if args["slices"] else slices_all
    # rectangle = False if args["slices"] else True

    slices_plot = slices_domain
    slices_plot = slices_training
    plot_velocity_density(flow.f, flow=flow, config=args, slices=slices_plot, title="simulation", rectangle=True)
    # plotRho(flow.f, flow=flow, config=args, slices=slices_plot, rectangle=rectangle)

    if args["load_dataset"]:
        if args["train"]:
            plot_velocity_density(reference, flow=flow, config=args, title="reference", rectangle=False)
        elif args["show_dataset_idx"] is not None:
            reference = dataset_train.get_f(args["show_dataset_idx"])
            plot_velocity_density(reference, flow=flow, config=args, slices=slices_plot, title="reference", rectangle=False)
            print("Difference: ", (reference - flow.f).sum())


    # if args["K_neural"]:
    #     print(K_tuned(
    #                   flow.f[:,slices[0].stop,:],
    #                   flow.boundaries[1].rho_dt_old,
    #                   flow.boundaries[1].u_dt_old,
    #                   flow.boundaries[1].v_dt_old))
    if args["train"]:
        plot = PlotNeuralNetwork(base="./", show=True, style="./ecostyle.mplstyle")
        plot.loss_function(np.array(epoch_training_loss)/epoch_training_loss[0], name=args["loss_plot_name"])
    if args["K_neural"]:
        print("K0 tuned max: ", K_tuned.K0max, "K0 tuned min: ", K_tuned.K0min)
        print("K1 tuned max: ", K_tuned.K1max, "K1 tuned min: ", K_tuned.K1min)


    if reporter is not None:
        out = torch.tensor(reporter.out_total).cpu().detach()
        t = torch.tensor(reporter.t).cpu().detach()
        plt.plot(t,out)
        plt.ylim(1-1e-5,1+1.3e-5)
        plt.show()
