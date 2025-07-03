# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

"""A class for the cluster sampling method"""
class Smpl():
    
    def __init__(self, data):
        self.data = data
    
    def cluster_sample(self, k=5):
        """

        Parameters
        ----------
        k : String, optional
            The default is 5. This is the nunber of clusters (Number of samples for DCI)

        Returns
        -------
        closest_pt_idx : List
            The closest datapoint index for each cluster centroid.
        centroids : numpy array
            Cluster centroids

        """
        kmeans_kwargs = {"init":"random", "n_init":10, "max_iter":300, "random_state":42}
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        labels= kmeans.fit_predict(self.data)
        centroids = kmeans.cluster_centers_
        labels = labels+1
        closest_pt_idx = []
        for iclust in range(kmeans.n_clusters):
            cluster_pts = self.data[kmeans.labels_ == iclust]
            cluster_pts_indices = np.where(kmeans.labels_ == iclust)[0]
            cluster_cen = kmeans.cluster_centers_[iclust]
            min_idx = np.argmin([euclidean(self.data[idx], cluster_cen) for idx in cluster_pts_indices])
            closest_pt_idx.append(cluster_pts_indices[min_idx])
        return closest_pt_idx, centroids
