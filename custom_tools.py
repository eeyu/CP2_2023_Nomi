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

def generate_random_grids(num_grids):
    # numpy_grids = np.random.randint(0, 5, size=(num_grids, 7, 7))
    p = np.array([1, 1, 1, 1, 1])
    # 3 likes industrial and 1 likes commercial, residential
    p = p / np.sum(p)
    return np.random.choice(np.arange(5), size = (num_grids,7,7), p=p) #Randomly Sample Grids
    # return numpy_grids

if __name__ == "__main__":
    print(generate_random_grids(5))

