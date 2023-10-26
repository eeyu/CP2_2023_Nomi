import os
import torch

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
PARAM_PATH = HOME_PATH + "parameters" + "/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 0
