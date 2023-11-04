import path
import execute.training_dataset as training
from torch.nn import Module
import torch
from typing import List
import numpy as np
import utils_public as up


class PredictionModel:
    def __init__(self):
        # Load up models
        self.models:List[Module] = []
        for i in range(4):
            favorite_file = path.get_favorite_parameters(i)
            print(favorite_file)
            model, a, b = training.load_model_from_file(favorite_file)
            self.models.append(model)

    def _get_model(self, i) -> Module:
        return self.models[i]

    # grids is n x 7 x 7 for n pieces of data
    # output: n x 4 predictions
    def do_prediction(self, grids):
        device = "cuda"
        x = training.get_one_hot(grids).type(torch.float)
        x = x.to(device)
        num_data = x.shape[0]
        preds = np.zeros((num_data, 4))
        with torch.no_grad():
            for i in range(4):
                model = self._get_model(i).to(device)
                model.eval()
                preds[:, i] = model(x).to("cpu")
        # print(preds.shape)
        return preds

    # input: n grids
    # print: number of valid grids
    def evaluate_num_valid_predictions(self, grids):
        num_grids = len(grids)
        num_splits = int(num_grids / 100000) + 1

        if num_splits > 1:
            split_x = self._split_grids(grids, num_splits)
            split_y = []
            for i in range(num_splits):
                split_y.append(self.do_prediction(split_x[i]))
            ratings = np.vstack(split_y)
        else:
            ratings = self.do_prediction(grids)

        min_predictions = np.min(ratings, axis=1)
        print(f"Number of valid grids (as predicted): {np.sum(min_predictions > 0.85)}")
        print(f"best predicted scores: {ratings[np.argmax(min_predictions)]}")
        up.plot_ratings_histogram(ratings)
        return ratings

    # (n, 7, 7)
    def _split_grids(self, x, num_splits):
        num_data = len(x)
        split_length = int(num_data / num_splits)
        start_index = 0
        split_x = []
        for i in range(num_splits - 1):
            split_x.append(x[start_index:start_index + split_length, :, :])
            start_index += split_length
        split_x.append(x[start_index:, :, :])
        return split_x

