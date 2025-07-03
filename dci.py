# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from plotting import plots
from sampling import Smpl

class DCI():
    """ Class for Dimensional Cophenetic Integrity (DCI)."""
    
    def __init__(self, embedding, data_parser=None, emb_name=None, percent=None):
        """

        Parameters
        ----------
        embedding : Numpy array
            A dimensionality reduced embedding of data where rows = samples, columns = features.
        data_parser : DataParser object, optional
            a dataparser object
        emb_name : String, optional
            The name of the embedding
        percent : Int, optional
            percentage of total samples to use for the 'cluster' sampling method
        """
       
        if data_parser is not None:
            self.data_parser = data_parser
            self.data = data_parser.data
            self.mzs = data_parser.mzs
            self.mask = data_parser.mask
            self.emb_name = emb_name
        
        self.embedding = embedding
        self.percent = percent
    
    
    def sample_distance(self, num_pixels=20, sampling="cluster", percentage=5):
        """

        Parameters
        ----------
        num_pixels : Int, optional
            Number of samples to use for the 'random' sampling method
        sampling : String, optional
            The sampling method for DCI
        percentage : Int, optional
            Percentage of total samples to use in the 'cluster' method

        Returns
        -------
        raw_distances : Numpy array
            Condensed pairwise distance matrix for the raw (high dimensional) data
        emb_distances : Numpy array
            Condensed pairwise distance matrix for the embedding (low dimensional) data
        random_pixels : Numpy array
            Sampled datapoints
        centroids : Numpy array
            Cluster centroids

        """

        if sampling == "random":
            rng = default_rng()
            random_pixels = rng.integers(low=0, high=len(self.data), size=num_pixels)
            centroids=None
            
        if sampling == "cluster":
            random_pixels,centroids = Smpl(self.embedding.astype(np.float64)).cluster_sample(k=int((len(self.embedding)/100)*percentage))
            sampled_data = self.embedding[random_pixels]
            if self.emb_name is None:
                print("DataParser object does not have the attribute: emb_name. Due to not loading an embedding or loading an embedding without a name")
                self.emb_name="Embedding data distribution"
            if self.embedding.shape[1] == 2:
                plots(self.embedding).sample_points(self.emb_name, "Sampled data", sampled_data, None, None, mode="2d")
                plots(self.embedding).embedding_scatter(self.emb_name, mode="2d")
            if self.embedding.shape[1] == 3:
                plots(self.embedding).sample_points(self.emb_name, "Sampled data", sampled_data, None, None, mode="3d")
                plots(self.embedding).embedding_scatter(self.emb_name, mode="3d")

        raw_pixels = [self.data[i,:] for i in random_pixels]
        emb_pixels = [self.embedding[i,:] for i in random_pixels]
        raw_distances = pdist(raw_pixels, metric="cosine")
        emb_distances = pdist(emb_pixels, metric="euclidean")

        return raw_distances, emb_distances, random_pixels, centroids
    
    def dci(self, raw_distances=None, emb_distances=None, random_pixels=None, centroids=None, threshold=False, sampling="cluster", percentage=5, num_pixels=None):
        """
  
        Parameters
        ----------
        raw_distances : Numpy array, optional
            Condensed pairwise distance matrix for the raw (high dimensional) data
        emb_distances : Numpy array, optional
            Condensed pairwise distance matrix for the embedding (low dimensional) data
        centroids : List, optional
            Centroids used in the 'cluster' sampling method if applicable
        threshold : Bool, optional
            Option for thresholding the embedding image between 5%-95% intensity
        sampling : String, optional
            Sampling method for DCI
        percentage : Int, optional
            Percentage total samples to use for the 'cluster' sampling method
        num_pixels : Int, optional
            Number of samples to use for the 'random' sampling method
  
        Returns
        -------
        mi : Float
            The Mutual information between high and low cophenetic distance matrices
        dci_dict : Dict
            Dictionary of DCI results and outputs
        emb_img : Numpy array
            A masked image array of the embedded data
        fig : Matplotlib object
            Cophenetic distance matrix heatmap
  
          """
          
        if random_pixels is None:
            raw_distances, emb_distances, random_pixels, centroids = self.sample_distance(sampling=sampling, percentage=percentage, num_pixels=num_pixels)
            
        if random_pixels is not None:
            raw_pixels = [self.data[i,:] for i in random_pixels]
            emb_pixels = [self.embedding[i,:] for i in random_pixels]
            raw_distances = pdist(raw_pixels, metric="cosine")
            emb_distances = pdist(emb_pixels, metric="euclidean")
        
        emb_img = None
        if self.embedding.shape[1]==3:
            if threshold is True:
                    im_threshold = plots(self.embedding).embedding_threshold(lower=5, upper=95)
                    im = plots(im_threshold)
                    emb_img, rgb_channels = im.gen_rgb_im(self.mask)
            if threshold is False:
                    im = plots(self.embedding)
                    emb_img, rgb_channels = im.gen_rgb_im(self.mask)
                    plt.imshow(emb_img)
                    plt.axis(False)
                    plt.title(self.emb_name)

        data_plot = plots(raw_distances, labels=random_pixels)
        data1_cophenetic_distance, data1_cophenetic_matrix, linkage_matrix1 = data_plot.agglomerative_clustering_dendrogram(metric="cosine")
                                                                                  
        emb_plot = plots(emb_distances, labels=random_pixels)
        data2_cophenetic_distance, data2_cophenetic_matrix, linkage_matrix2 = emb_plot.agglomerative_clustering_dendrogram(metric="euclidean")

        data1_cophenetic_matrix = squareform(data1_cophenetic_matrix)
        data2_cophenetic_matrix = squareform(data2_cophenetic_matrix)
        
        def heatmap_fig():
            print("Generating cophenetic distance matrices")
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            data_plot = plots(data1_cophenetic_matrix)
            emb_plot = plots(data2_cophenetic_matrix)
            dataheatmap = data_plot.heatmap("High dimensional data pixels", xtick=random_pixels, ytick=random_pixels, ax=axes[0])
            embheatmap = emb_plot.heatmap((self.emb_name+" data pixels"), xtick=random_pixels, ytick=random_pixels, ax=axes[1])
            plt.show()
            plt.rcdefaults()
            return fig
        fig = heatmap_fig()
        
        mi = normalized_mutual_info_score(squareform(data1_cophenetic_matrix), squareform(data2_cophenetic_matrix))
        print(mi, " DCI score")
        dci_dict = {}
        dci_dict["data1_cophenetic_matrix"] = data1_cophenetic_matrix
        dci_dict["data2_cophenetic_matrix"] = data2_cophenetic_matrix
        dci_dict["random_pixels"] = random_pixels
        dci_dict["linkage_matrix1"] = linkage_matrix1
        dci_dict["linkage_matrix2"] = linkage_matrix2
        dci_dict["centroids"] = centroids
        dci_dict["DCI"] = mi

        return mi, dci_dict, emb_img, fig