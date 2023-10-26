from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import functional as F
from dataclasses import dataclass
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import custom_tools as ct
import path
from regression_model import ModelParameters

import utils_public as ut
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
def train(model : Module, trainDataLoader, valDataLoader, opt, epochs, run_name = ""):
    device = path.device
    lossFn = nn.MSELoss()
    num_steps_without_improvement = 0
    max_steps_no_improvement = 20
    best_validation_loss = 100 # large number
    best_parameters = model.state_dict()

    # loop over our epochs
    for e in range(0, epochs):
        print("epoch: " + str(e) + "/" + str(epochs) + " | " + "run: " + str(run_name) + " | " + "no imp: " + str(num_steps_without_improvement))
        # set the model in training mode
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        # loop over the training set
        for (x, y) in trainDataLoader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model.forward(x)
            loss = lossFn(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            totalTrainLoss += loss.item() / len(trainDataLoader.dataset)
        print("- train loss: \t" + "{:.5e}".format(totalTrainLoss))

        # validation step
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            totalValLoss = 0
            # loop over the validation set
            for (x, y) in valDataLoader:
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                loss = lossFn(pred, y)
                totalValLoss += loss.item() / len(valDataLoader.dataset)
            print("- val loss: \t" + "{:.5e}".format(totalValLoss))

        # Exit if validation is not improving
        if totalValLoss < best_validation_loss:
            best_parameters = model.state_dict()
            num_steps_without_improvement = 0
            best_validation_loss = totalValLoss
        else:
            num_steps_without_improvement += 1

        if num_steps_without_improvement >= max_steps_no_improvement:
            break

    model.load_state_dict(best_parameters)
    return model.state_dict(), best_validation_loss

def get_one_hot(grids):
    grids_oh = F.one_hot(torch.tensor(grids).type(torch.int64)).permute(0, 3, 1, 2) #[data, x1, x2, encode] -> [data, encode, x1, x2]
    # grids_oh = (np.arange(5) == grids[..., None]).astype(float)
    return grids_oh

# This one-hot encodes the labelled data for a given advisor
class AdvisorDataset(Dataset):
    def __init__(self, advisor):
        grids = ut.load_grids()  # Helper function we have provided to load the grids from the dataset
        ratings = ct.get_ratings()

        # gets subset of the dataset rated by advisor
        grids_subset, ratings_subset = ut.select_rated_subset(grids, ratings[:, advisor])
        grids_oh = get_one_hot(grids_subset)
        # grids_oh = F.one_hot(torch.from_numpy(grids_subset).type(torch.int64))
        # self.x = grids_oh
        self.x = grids_oh.type(torch.float)
        self.y = torch.tensor(ratings_subset).type(torch.float)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



class GenerateDataloader:
    def __init__(self, dataset: Dataset, batch_size, nval=0.1, ntest=0.1 ):
        (trainData, valData, test) = self.get_splits(dataset, n_val=nval, n_test=ntest)
        self.trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=batch_size)
        self.valDataLoader = DataLoader(valData, batch_size=batch_size)
        self.testDataLoader = DataLoader(test, batch_size=batch_size)

    # TODO need to make splits consistent when run across different files
    def get_splits(self, dataset: Dataset, n_val=0.1, n_test=0.1):
        length = dataset.__len__()
        # Determine sizes
        test_size = round(n_test * length)
        val_size = round(n_val * length)
        train_size = length - test_size - val_size
        # Calculate the split
        (trainData, valData, testData) = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return trainData, valData, testData

def save_model(model : Module, name):
    torch.save(model.state_dict(), path.PARAM_PATH + "_" + name)

def load_model(model : Module, name):
    model.load_state_dict(torch.load(path.PARAM_PATH + "_" + name))
    model.eval()

def save_model_and_hp(model : Module, hyperparam : ModelParameters, batch_size, advisor, name):
    dict = {}
    dict["param"] = model.state_dict()
    dict["hyperparam"] = hyperparam
    dict["batch_size"] = batch_size
    dict["advisor"] = advisor
    with open(path.PARAM_PATH + "_" + name, 'wb') as handle:
        pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model_from_file():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(initialdir="../parameters", defaultextension="pickle")  # show an "Open" dialog box and return the path to the selected file
    with open(filename, 'rb') as handle:
        dict = pickle.load(handle)
        hyperparam : ModelParameters = dict["hyperparam"]
        param = dict["param"]
        batch_size = dict["batch_size"]

        model = hyperparam.instantiate_new_model()
        model.load_state_dict(param)
        return model, batch_size, dict["advisor"]

def get_save_name(advisor, modelName, algName, run_index, extension):
    return "A" + str(advisor) + "_" + "R" + str(run_index) + "_" + modelName + "_" + algName  + extension
