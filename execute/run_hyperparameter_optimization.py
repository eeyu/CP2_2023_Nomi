from utils_public import *
import numpy as np
import train_dataset as train
import pandas as pd
from regression_model import TestNet, TestNetParameters, ModelParameters
import math

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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

advisor = 2
NUM_TRIALS = 50
run_index = 0
BATCH_SIZE = 256

def convert_config_to_param(config : Configuration, modelParam : ModelParameters):
    modelParam.set_from_configuration(config)

    # Run optimization
    optParam = AdamOptimizationParameters(
        INIT_LR=config["INIT_LR"],
        WEIGHT_DECAY=config["WEIGHT_DECAY"],
        # BATCH_SIZE= int(math.pow(2, config["BATCH_SIZE_EXP"])),
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=300)
    return modelParam, optParam


iteration = 0
dataset = train.AdvisorDataset(advisor)
dataloader = train.GenerateDataloader(dataset=dataset, batch_size=BATCH_SIZE, nval=0.15, ntest=0.15)
def train_loop(config : Configuration, seed : int=0) -> float:
    print("=============================")
    print("=============================")
    # Hyperparameters
    layout, optParam = convert_config_to_param(config)

    model = layout.instantiate_new_model()
    opt = Adam(model.parameters(), lr=optParam.INIT_LR, weight_decay=optParam.WEIGHT_DECAY)
    global dataloader
    train_dataloader = dataloader.trainDataLoader
    val_dataloader = dataloader.valDataLoader

    global iteration
    run_name = str(iteration) + "/" + str(NUM_TRIALS)
    param, val_loss = train.train(model=model, opt=opt, trainDataLoader=train_dataloader, valDataLoader=val_dataloader, epochs=optParam.EPOCHS, run_name=run_name)
    iteration += 1
    return val_loss


configspace_model = TestNetParameters.get_configuration_space()
configspace_alg = AdamOptimizationParameters.get_configuration_space()
def merge_config_space(configspace1 : ConfigurationSpace, configspace2 : ConfigurationSpace):
    configspace1.add_hyperparameters(configspace2._hyperparameters.values())
    return configspace1


configspace = merge_config_space(configspace_model, configspace_alg)

# Scenario object specifying the optimization environment
scenario = Scenario(configspace, deterministic=True, n_trials=NUM_TRIALS, n_workers=1)

# Use SMAC to find the best configuration/hyperparameters
smac = HyperparameterOptimizationFacade(scenario, train_loop)
incumbent = smac.optimize()

print("===========================================")
print("======= DONE! TRAINING INCUMBENT =========")
print("===========================================")
# Train on best parameters
layout, optParam = convert_config_to_param(incumbent)
model = layout.instantiate_new_model()
opt = Adam(model.parameters(), lr=optParam.INIT_LR, weight_decay=optParam.WEIGHT_DECAY)
dataset = train.AdvisorDataset(advisor)
dataloader = train.GenerateDataloader(dataset=dataset, batch_size=optParam.BATCH_SIZE, nval=0.15, ntest=0.15)
train_dataloader = dataloader.trainDataLoader
val_dataloader = dataloader.valDataLoader

train.train(model=model, opt=opt, trainDataLoader=train_dataloader, valDataLoader=val_dataloader,
            epochs=optParam.EPOCHS)

# Save training from best parameters
save_name = train.get_save_name(advisor=advisor, modelName=layout.getName(), algName=optParam.getName(), run_index="SMAC" + str(run_index), extension=".pt")
train.save_model(model, save_name)