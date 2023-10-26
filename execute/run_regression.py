from utils_public import *
import numpy as np
import train_dataset as train
import pandas as pd
from regression_model import TestNet, TestNetParameters, WideNetParameters

from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
import pickle
from OptimizationParameters import AdamOptimizationParameters

# Hyperparameters
advisor = 2
# layout = TestNetParameters(
#     pad1=1,
#     channel1=5,
#     pad2=1,
#     channel2=5,
#     fc2=10) # encode this layout
layout = WideNetParameters(
    pad1=1,
    channel1=5,
    width_pad1=2,
    pad2=1,
    channel2=5,
    width_pad2=1,
    fc2=10,
    fc3=10) # encode this layout
run_index = "TEST_WIDE"

# Run optimization
optParam = AdamOptimizationParameters(
    INIT_LR=1.324e-4,
    WEIGHT_DECAY=1.5983e-5,
    BATCH_SIZE=64,
    EPOCHS=50)


# model = TestNet(layout)
model = layout.instantiate_new_model()
opt = Adam(model.parameters(), lr=optParam.INIT_LR, weight_decay=optParam.WEIGHT_DECAY)
dataset = train.AdvisorDataset(advisor)
dataloader = train.GenerateDataloader(dataset=dataset, batch_size=optParam.BATCH_SIZE, nval=0.15, ntest=0.15)
train_dataloader = dataloader.trainDataLoader
val_dataloader = dataloader.valDataLoader

train.train(model=model, opt=opt, trainDataLoader=train_dataloader, valDataLoader=val_dataloader, epochs=optParam.EPOCHS)
save_name = train.get_save_name(advisor=advisor, modelName=layout.getName(), algName=optParam.getName(), run_index=run_index, extension=".pickle")
# train.save_model(model, save_name)
train.save_model_and_hp(model, layout, batch_size=optParam.BATCH_SIZE, name=save_name)