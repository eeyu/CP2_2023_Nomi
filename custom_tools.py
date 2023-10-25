from utils_public import *
import numpy as np
import pandas as pd
import path
def get_ratings():
    return np.load(path.HOME_PATH + "datasets/scores.npy")  # Load advisor scores

def get_ratings_as_df(ratings):
    score_order = ["Wellness", "Tax", "Transportation", "Business"]  # This is the order of the scores in the dataset
    ratings_df = pd.DataFrame(ratings, columns=score_order)  # Create a dataframe
    return ratings_df

def get_one_hot(grids):
    grids_oh = (np.arange(5) == grids[..., None]).astype(float)
    return grids_oh

