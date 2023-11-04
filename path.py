import os
import torch

HOME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
PARAM_PATH = HOME_PATH + "parameters" + "/"
def get_favorite_parameters(advisor: int):
    path = HOME_PATH + "favorite_parameters/" + str(advisor) + "/"
    contents = os.listdir(path)[0]
    return path + contents

def get_ga_output_name(name, extension):
    path = HOME_PATH + "ga_outputs/"
    return path + name + extension

def get_valid_grids_name(name, extension):
    path = HOME_PATH + "valid_predicted_grids/"
    return path + name + extension

def get_submission_name(name, id, extension):
    path = HOME_PATH + "submission/" + name + "/"
    if not os.path.isdir(path):
        os.mkdir(path)
    return path + str(id) + extension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
seed = 0
