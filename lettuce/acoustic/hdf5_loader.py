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

context = lt.Context(torch.device("cuda:0"), use_native=False, dtype=torch.float64)
dataset_train = LettuceDataset(context=context, filebase="dataset_mach-0.30_interv-0.25.h5", target=False)
print(dataset_train)
print(dataset_train)