# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, cophenet
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler

class plots():
    
    def __init__(self, data, labels=None):
        """

        Parameters
        ----------
        data : Numpy array
            data is in the form of a condensed distance matrix, or 2/3D embedding
        labels : Numpy array
            Cluster labels
            
        """
        
        self.data = data
        self.labels = labels
        
    def heatmap(self, title, xtick=None, ytick=None, ax=None):
        """

        Parameters
        ----------
        title : String, 
            Heatmap title

        Returns
        -------
        ax : Axes object
            Heatmap figure
        
        """
        if ax is None:
            print("Axes cannot be None")
            
        decimal_format = FuncFormatter(lambda x, _:"{:.1e}".format(x))
        sns.heatmap(self.data, annot=False, cmap="viridis", fmt=".1e", ax=ax, xticklabels=xtick, yticklabels=ytick, cbar_kws={'format':decimal_format},
                    mask = np.eye(len(self.data)))
        ax.set_title(title)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(False)
        return ax
        
    def agglomerative_clustering_dendrogram(self, metric="euclidean"):
        """

        Parameters
        ----------
        metric : String, 
            The metric for agglomerative/ hierarchical clustering

        Returns
        -------
        cophenetic_distance : Numpy array
            Heatmap figure
        cophenetic_matrix : Numpy array
            Cophenetic correlation distance
        linkage_matrix : Numpy array
            Condensed cophenetic distance matrix
        
        """
        linkage_matrix = linkage(self.data, method='average', metric=metric)
        cophenetic_distance, cophenetic_matrix = cophenet(linkage_matrix, self.data)
        return cophenetic_distance, cophenetic_matrix, linkage_matrix
    
    def gen_rgb_im(self, mask):
        """

        Parameters
        ----------
        mask : Bool numpy array, 
            Bool mask for converting a 3D embedding into an image

        Returns
        -------
        rgb_im : Numpy array
            Image array
        rgb_channels : Numpy array
            3D numpy array
        """
        load_emb = self.data
        
        def norm_im(im):
            im = (im-np.min(im))/(np.max(im)-np.min(im))
            return im
        
        R = norm_im(load_emb[:,0])
        G = norm_im(load_emb[:,1])
        B = norm_im(load_emb[:,2])
        
        def get_embedded_im(chan, mask):
            im = np.zeros(mask.size)
            im[mask.ravel()] =  chan
            im = im.reshape((mask.shape[0], mask.shape[1]))
            im[im==0] = 1
            return im
        
        R = get_embedded_im(R, mask)
        G = get_embedded_im(G, mask)
        B = get_embedded_im(B, mask)
        
        rgb_im = np.dstack((R,G,B))
        rgb_mask = np.stack((mask, mask, mask),axis=2)
        rgb_masked = rgb_mask*rgb_im
        
        rgb_channels = np.vstack(rgb_masked)
        return rgb_im, rgb_channels

    def embedding_threshold(self, lower=None, upper=None):
        if lower is None:
            Dullest = self.data.min()
        else:
            Dullest = np.percentile(self.data, lower, axis=0)
        Threshold = np.percentile(self.data, upper, axis=0)
        threshold_data = np.clip(self.data, Dullest, Threshold)
        return threshold_data
    
    def sample_points(self, title, title1, sample1, title2=None, sample2=None, mode="3d"):
        """Plotting original data and sampled data points 3D"""
        if mode == "3d":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c='gray', alpha=0.01, s=10, label='Original Data')
            ax.scatter(sample1[:, 0], sample1[:, 1], sample1[:, 2], c='red', alpha=0.7, s=10, label=title1)
            if title2 is not None:
                ax.scatter(sample2[:, 0], sample2[:, 1], sample2[:, 2], c='blue', alpha=0.7, s=10, label=title2)
            plt.legend()
            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_zlabel('Blue')
            
        if mode == "2d":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self.data[:, 0], self.data[:, 1], c='gray', alpha=0.01, s=10, label='Original Data')
            ax.scatter(sample1[:, 0], sample1[:, 1], c='red', alpha=0.7, s=10, label=title1)
            if title2 is not None:
                ax.scatter(sample2[:, 0], sample2[:, 1], c='blue', alpha=0.7, s=10, label=title2)
                plt.legend()
                ax.set_xlabel('1st dimension')
                ax.set_ylabel('2nd dimension')
                
        plt.title(title)
        plt.show()
        
    def embedding_scatter(self, title, mode="3d", labels=None, s=1):
        """Plotting original data and sampled data points 3D"""
        if mode == "3d":
            scaler = MinMaxScaler()
            colours = scaler.fit_transform(self.data)
            colours=np.clip(colours, 0, 1)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=colours, s=10)
            plt.legend()
            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_zlabel('Blue')
            
        if mode == "2d":
            if labels is not None:
                colours = ['red', 'green', 'blue', 'orange', 'cyan', 'pink', 'magenta', 'maroon', 'black']
                c = [colours[label] for label in labels-1]
                patches = [mpatches.Patch(color=colours[i], label=f'Cluster {i+1}') for i in range(len(np.unique(labels)))]
            else:
                c = 'blue'
            scaler = MinMaxScaler()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(self.data[:, 0], self.data[:, 1], c=c, s=s)
            if labels is not None:
                ax.legend(handles=patches, title="Cluster", loc='upper right', bbox_to_anchor=(1.3, 1))
            ax.set_xlabel('1st dimension')
            ax.set_ylabel('2nd dimension')

        plt.title(title)
        plt.grid(False)
        plt.show()