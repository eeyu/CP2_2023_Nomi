from utils_public import *
import numpy as np
import pandas as pd

# Loads grids that are provided
grids = load_grids() #Helper function we have provided to load the grids from the dataset
grids.shape #Check shape
# 500000 x 7 x 7

# Provided advisor scores. only 1% are labelled (5000)
ratings = np.load("datasets/scores.npy") #Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe
display(ratings_df) #Print dataframe

# Obtains the 5000 rated datasets
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,0]) #gets subset of the dataset rated by advisor 0
print(grids_subset.shape)
print(ratings_subset.shape)

# Evaluate diversity
diversity_score(grids[:100])

# Goal: train regressor on the 5000 data sets.

# Obtains the predictions
def get_predictions(grids, ratings, predictor):
    grids = grids.reshape(grids.shape[0], 49)
    grids_df = pd.DataFrame(grids, columns = range(grids.shape[1]))
    predictions = predictor.predict(grids_df).values
    mask = np.where(~np.isnan(ratings))
    predictions[mask] = ratings[mask]
    return predictions
predictions = get_predictions(grids, ratings[:,0], predictor)