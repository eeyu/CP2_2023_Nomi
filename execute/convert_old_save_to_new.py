import path
from utils_public import *
import numpy as np
import train_dataset as train
import pandas as pd
from regression_model import TestNet, TestNetParameters

from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Module
import argparse
import torch
import time
from regression_model import ModelParameters

import utils_public as ut
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import pickle
from OptimizationParameters import AdamOptimizationParameters
advisor = 2

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(initialdir="../parameters",
                           defaultextension="pickle")  # show an "Open" dialog box and return the path to the selected file
with open(filename, 'rb') as handle:
    dict = pickle.load(handle)
    hyperparam: ModelParameters = dict["hyperparam"]
    param = dict["param"]
    batch_size = dict["batch_size"]

    model = hyperparam.instantiate_new_model()
    model.load_state_dict(param)

    dict["advisor"] = advisor
    with open(filename, 'wb') as handle1:
        pickle.dump(dict, handle1, protocol=pickle.HIGHEST_PROTOCOL)

# advisor = 0
#
# if advisor == 0:  # 0.885
#     layout = TestNetParameters(
#         pad1=1,
#         channel1=14,
#         pad2=1,
#         channel2=2,
#         fc2=14)  # encode this layout
#
#     optParam = AdamOptimizationParameters(
#         INIT_LR=7.9e-4,
#         WEIGHT_DECAY=9.2e-4,
#         BATCH_SIZE=64,
#         EPOCHS=247)
#     run_index = "SMAC"
#
# elif advisor == 1:  # GOOD!! 0.99
#     layout = TestNetParameters(
#         pad1=1,
#         channel1=5,
#         pad2=1,
#         channel2=5,
#         fc2=10)  # encode this layout
#
#     optParam = AdamOptimizationParameters(
#         INIT_LR=5.0e-4,
#         WEIGHT_DECAY=5.0e-5,
#         BATCH_SIZE=64,
#         EPOCHS=100)
#     run_index = "0"
#
# elif advisor == 2:  # 0.83
#     layout = TestNetParameters(
#         pad1=1,
#         channel1=16,
#         pad2=1,
#         channel2=20,
#         fc2=7)  # encode this layout
#
#     optParam = AdamOptimizationParameters(
#         INIT_LR=2.1e-03,
#         WEIGHT_DECAY=3.6e-03,
#         BATCH_SIZE=256,
#         EPOCHS=300)
#     run_index = "SMAC0"
# else:  # 3 0.967
#     layout = TestNetParameters(
#         pad1=3,
#         channel1=14,
#         pad2=2,
#         channel2=20,
#         fc2=7)  # encode this layout
#
#     optParam = AdamOptimizationParameters(
#         INIT_LR=5.1e-3,
#         WEIGHT_DECAY=8.5e-4,
#         BATCH_SIZE=256,
#         EPOCHS=300)
#     run_index = "SMAC0"
#
# # Value in file
# batch_size = optParam.BATCH_SIZE
# old_save_name = train.get_save_name(advisor=advisor, modelName=layout.getName(), algName=optParam.getName(),
#                                 run_index=run_index, extension=".pt")
#
# model = layout.instantiate_new_model()
# train.load_model(model, old_save_name)
#
# # New save
# new_save_name = train.get_save_name(advisor=advisor, modelName=layout.getName(), algName=optParam.getName(),
#                                 run_index=run_index, extension=".pickle")
# train.save_model_and_hp(model, layout, batch_size, name=new_save_name)