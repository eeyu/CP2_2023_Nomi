# take inputs in the original format, converts one-hot, and feeds to trained model

import utils_public as up
import numpy as np
from execute.trained_predictor import PredictionModel
import run_generate_grids_GA as generate
import path
import pickle
from run_generate_grids_GA import GenerateGrids

def save_valid_grids(name, good_grids, good_ratings):
    save_name = path.get_valid_grids_name(name=name, extension=".pickle")
    with open(save_name, 'wb') as f:
        pickle.dump(save, f)

def load_valid_grids(name):
    save_name = path.get_valid_grids_name(name=name, extension=".pickle")
    with open(save_name, 'rb') as f:
        output = pickle.load(f)
    return output["good_grids"], output["good_ratings"]

if __name__ == "__main__":
    prediction_model = PredictionModel()

    grids = up.load_grids()
    # Load more grids generated from GA
    more_grids = generate.load_ga_grids_from_file("GA_0.9")
    # grids = np.concatenate([grids, more_grids])
    grids = more_grids

    # Obtain valid grids from prediction
    ratings = prediction_model.evaluate_num_valid_predictions(grids)
    min_predictions = np.min(ratings, axis=1)

    if len(grids) == len(min_predictions):
        good_grids = grids
        good_ratings = ratings
    else:
        good_grids = grids[min_predictions > 0.85]
        good_ratings = ratings[min_predictions > 0.85]

    save = {
        "good_grids": good_grids,
        "good_ratings": good_ratings
    }

    # print(good_grids.shape)
    # print(good_ratings.shape)

    # Save those grids
    save_name = path.get_valid_grids_name(name="GA_0.9", extension=".pickle")
    with open(save_name, 'wb') as f:
        pickle.dump(save, f)


    # top_100_indices = np.argpartition(min_predictions, -100)[-100:]  # indices of top 100 designs (as sorted by minimum advisor score)
    # final_submission = grids[top_100_indices].astype(int)
    # assert final_submission.shape == (100, 7, 7)
    # assert final_submission.dtype == int
    # assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))

    # save all valid scores

    # id = np.random.randint(1e8, 1e9 - 1)
    # np.save(f"{id}.npy", final_submission)

