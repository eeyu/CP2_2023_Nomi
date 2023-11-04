from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Integer
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import pickle
import path
from trained_predictor import PredictionModel
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.indicators.hv import HV


class GenerateGrids(ElementwiseProblem):
    def __init__(self, regression_model: PredictionModel):
        self.regression_model = regression_model
        variables = dict()
        self.min_score = 0.9

        # Our position matrix consists of Nx2 real numbers (cartesian coordinate values) between 0 and 1
        for i in range(7):
            for j in range(7):
                variables["M" + str(i) + str(j)] = Integer(bounds=(0, 4))

        # Set up some variables in the problem class we inherit for pymoo
        # n_obj=number of objectives, n_constr=number of constraints
        # Our objectives are chamfer distance and material, and they both have constraints.
        super().__init__(vars=variables, n_obj=4, n_constr=4)

    @staticmethod
    def instantiate_from_parameters(x):
        grid = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                grid[i, j] = x["M" + str(i) + str(j)]
        return grid

    def _evaluate(self, x, out, *args, **kwargs):
        # Convert to mechanism representation
        grid = GenerateGrids.instantiate_from_parameters(x)

        # Call our evaluate function to get validity, CD, and material use
        scores = self.regression_model.do_prediction([grid])[0]

        valid = np.min(scores) > self.min_score

        # check to see if the mechanism is valid
        if not valid:
            # if mechanism is invalid set the objective to infinity
            out["F"] = [np.Inf, np.Inf, np.Inf, np.Inf]
        else:
            out["F"] = [scores[0],
                        scores[1],
                        scores[2],
                        scores[3]]
        out["G"] = [-(scores[0] - self.min_score),
                    -(scores[1] - self.min_score),
                    -(scores[2] - self.min_score),
                    -(scores[3] - self.min_score)]


def plot_HV(F, ref):
    # Plot the designs
    plt.scatter(F[:, 1], F[:, 0])

    # plot the reference point
    plt.scatter(ref[1], ref[0], color="red")

    # plot labels
    plt.xlabel('Material Use')
    plt.ylabel('Chamfer Distance')

    # sort designs and append reference point
    sorted_performance = F[np.argsort(F[:, 1])]
    sorted_performance = np.concatenate([sorted_performance, [ref]])

    # create "ghost points" for inner corners
    inner_corners = np.stack([sorted_performance[:, 0], np.roll(sorted_performance[:, 1], -1)]).T

    # Interleave designs and ghost points
    final = np.empty((sorted_performance.shape[0] * 2, 2))
    final[::2, :] = sorted_performance
    final[1::2, :] = inner_corners

    # Create filled polygon
    plt.fill(final[:, 1], final[:, 0], color="#008cff", alpha=0.2)


def load_grids_from_results(results):
    grids = []
    for x in results.pop.get("X"):
        grids.append(GenerateGrids.instantiate_from_parameters(x))
    print("N grids loaded: " + str(len(grids)))
    return grids


def save_ga_results(results, name):
    save_name = path.get_ga_output_name(name=name, extension=".pickle")
    with open(save_name, 'wb') as f:
        pickle.dump(results, f)


def load_ga_grids_from_file(name):
    save_name = path.get_ga_output_name(name=name, extension=".pickle")
    with open(save_name, 'rb') as f:
        grids_encoded = pickle.load(f)
    return load_grids_from_results(grids_encoded)


def analyze_ga_result(grids, prediction_model: PredictionModel):
    prediction_model.evaluate_num_valid_predictions(grids)

    # Specify reference point
    ref_point = np.array([0.1, 10])

    # Calculate Hypervolume
    ind = HV(ref_point)
    hypervolume = ind(grids.F)

    # Print and plot
    print('Hyper Volume ~ %f' % (hypervolume))
    plot_HV(grids.F, ref_point)
    plt.show()


if __name__ == "__main__":
    regression_model = PredictionModel()
    # Setup Problem
    problem = GenerateGrids(regression_model)

    # Set up GA with pop size of 100 -- see pymoo docs for more info on these settings!
    algorithm = NSGA2(pop_size=500, sampling=MixedVariableSampling(),
                      mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                      crossover=SBX(eta=0.2, prob=0.9),
                      mutation=GaussianMutation(sigma=1.0),
                      survival=RankAndCrowdingSurvival(),
                      eliminate_duplicates=MixedVariableDuplicateElimination())

    n_gen = 300
    results = minimize(problem,
                       algorithm,
                       ('n_gen', n_gen),
                       verbose=True,
                       save_history=True,
                       seed=path.seed)
    grids = load_grids_from_results(results)

    # save the results
    save_name = path.get_ga_output_name(name="GA_0.9", extension=".pickle")
    with open(save_name, 'wb') as f:
        pickle.dump(results, f)

    # grids = load_ga_grids_from_file("test1")

    analyze_ga_result(grids, regression_model)
