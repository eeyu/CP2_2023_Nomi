import torch

from utils_public import *
import numpy as np
import pandas as pd
import custom_tools as ct
import torch.nn.functional as F

# Loads grids that are provided
grids = load_grids() #Helper function we have provided to load the grids from the dataset
grids.shape #Check shape
# 500000 x 7 x 7

# Provided advisor scores. only 1% are labelled (5000)
ratings = ct.get_ratings()

# Obtains the 5000 rated datasets
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,0]) #gets subset of the dataset rated by advisor 0

a = torch.from_numpy(grids_subset[0]).type(torch.int64)
# grids_oh = (np.arange(5) == a[..., None]).astype(int)
print(a)
grids_oh = F.one_hot(a).permute(2, 0, 1)
print(grids_oh)