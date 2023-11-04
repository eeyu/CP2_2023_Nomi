import path
import pickle
import run_predict_valid_grids
import math
from abc import ABC, abstractmethod
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, BisectingKMeans
import numpy as np
import utils_public as up

class Clustering(ABC):
    @abstractmethod
    def get_labels(self, X, num_clusters):
        pass

class Ward(Clustering):
    def get_labels(self, X, num_clusters):
        ward = AgglomerativeClustering(
            n_clusters=num_clusters, linkage="ward"
        )
        ward.fit(X)
        return ward.labels_

class Spectral(Clustering):
    def get_labels(self, X, num_clusters):
        cluster = (SpectralClustering(n_clusters=num_clusters,
            assign_labels='cluster_qr',
            random_state=1121))
        cluster.fit(X)
        return cluster.labels_

class BKMeans(Clustering):
    def get_labels(self, X, num_clusters):
        cluster = BisectingKMeans(n_clusters=num_clusters, random_state=3)
        cluster.fit(X)
        return cluster.labels_

if __name__ == "__main__":
    seed = 4323
    np.random.seed(seed)


    default_grids, default_ratings = run_predict_valid_grids.load_valid_grids("orig")
    more_grids, more_ratings = run_predict_valid_grids.load_valid_grids("GA_0.9_max")
    more_grids = np.array(more_grids)

    num_default_grids = len(default_ratings)
    num_more_grids = len(more_ratings)
    num_grids_to_sample = 100 - num_default_grids
    print("To sample: " + str(num_grids_to_sample))

    # Now cluster
    # for n clusters
    # select the random (100 / n)^[round up] values from each cluster
    # take the top 100 of those values
    # save as submission
    # Cluster the data
    num_clusters = 10
    num_samples_per_cluster = math.ceil(num_grids_to_sample * 1.0 / num_clusters)

    flattened_grids = []
    for grid in more_grids:
        flattened_grids.append(grid.reshape((49)))

    clustering = Ward()
    labels = clustering.get_labels(X=flattened_grids, num_clusters=num_clusters)

    # Sample grids from each cluster
    indices_to_submit = []
    for label in range(num_clusters):
        indices = np.array(list(range(num_more_grids)))
        indices_for_label = indices[labels==label]
        print(indices_for_label)
        random_sampled_indices = np.random.choice(indices_for_label, size=num_samples_per_cluster, replace=False)
        indices_to_submit += random_sampled_indices.tolist()

    indices_to_submit = indices_to_submit[0:num_grids_to_sample]
    # indices_to_submit = np.random.choice(list(range(num_more_grids)), size=num_grids_to_sample, replace=False)
    # Obtain the grids and add to original list
    grids = default_grids
    print(indices_to_submit)
    more_grids_to_submit = more_grids[indices_to_submit, :, :]
    grids = np.concatenate([grids, more_grids_to_submit])

    final_submission = grids.astype(int)
    assert final_submission.shape == (100, 7, 7)
    assert final_submission.dtype == int
    assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))

    print(up.diversity_score(grids))

    # save all valid scores
    folder_name = "random"
    id = np.random.randint(1e8, 1e9 - 1)
    save_name = path.get_submission_name(folder_name, id, ".npy")
    np.save(save_name, final_submission)

