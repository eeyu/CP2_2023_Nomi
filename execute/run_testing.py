import path
from utils_public import *
import train_dataset as train

import matplotlib.pyplot as plt
from torch.nn import Module
import torch
from sklearn.metrics import r2_score
torch.manual_seed(path.seed)


# Hyperparameters
# advisor = 1

model, batch_size, advisor = train.load_model_from_file()
# advisor == 0: 0.885 / 0.844
# advisor == 1: GOOD!! 0.98 / 0.97
# advisor == 2: 0.91 / 0.81 (R4)
# advisor == 3: GOOD!! 0.967 / 0.95



dataset = train.AdvisorDataset(advisor)
dataloader = train.GenerateDataloader(dataset=dataset, batch_size=batch_size, nval=0.15, ntest=0.25)


def get_predictions_actual_for_data(model : Module, dataloader):
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        actual = []
        # loop over the test set
        for (x, y) in dataloader:
            # send the input to the device
            x = x.to(path.device)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.numpy())
            actual.extend(y.numpy())

        return preds, actual

preds_test, act_test = get_predictions_actual_for_data(model, dataloader.testDataLoader)
preds_train, act_train = get_predictions_actual_for_data(model, dataloader.trainDataLoader)

def plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i):
    print(f"Train Set R2 score: {r2_score(ratings_train, preds_train)}") #Calculate R2 score
    print(f"Test Set R2 score: {r2_score(ratings_test, preds_test)}")

    plt.scatter(ratings_train, preds_train, label='Train Set Preds', s=3, c = "#F08E18") #Train set in orange
    plt.scatter(ratings_test, preds_test, label='Test Set Preds', s=5, c = "#DC267F") #Test set in magenta
    plt.plot([0,1], [0,1], label="target", linewidth=3, c="k") # Target line in Black
    ax = plt.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Advisor {i} Predictions")
    plt.legend()
    plt.show()


plot_and_r2(preds_train, preds_test, act_train, act_test, advisor)