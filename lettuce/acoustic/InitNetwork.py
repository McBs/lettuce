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
# from plot import *
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import torch.optim as optim


class NeuralTuning(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device='cuda', nodes=20, index=None):
        """Initialize a neural network boundary model."""
        super(NeuralTuning, self).__init__()
        self.moment = D2Q9Dellar(lt.D2Q9(), lt.Context(device="cuda", dtype=torch.float64, use_native=False))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, nodes, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(nodes, 1, bias=True),
            torch.nn.Sigmoid(),
        ).to(dtype=dtype, device=device)
        self.index = index
        print("Initialized NeuralTuning")

    def forward(self, f):
        """Forward pass through the network with residual connection."""

        return self.net(f[:,25,:].transpose(0,1))


if __name__ == "__main__":
    K_tuned = NeuralTuning()
    context = lt.Context(torch.device("cuda:0"), use_native=False, dtype=torch.float64)
    flow = lt.TaylorGreenVortex(context, resolution=[100,100], reynolds_number=300, mach_number=0.3)
    # print(flow.f)
    K = K_tuned(flow.f)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(K_tuned.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    # optimizer = torch.optim.Adam(K_tuned.parameters(), lr=1e-2)
    epoch_training_loss = []
    optimizer.zero_grad()

    running_loss = []
    for i in range(2000):
        optimizer.zero_grad()
        k = K_tuned(flow.f)
        loss = criterion(k, torch.zeros_like(k))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        # print("running_loss:", running_loss)
    plt.plot(running_loss)
    plt.show()
    print(K_tuned(flow.f))
    print(running_loss[-1])
    torch.save(K_tuned, "model_training_v1.pt")