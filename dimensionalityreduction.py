# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""

import time
from sklearn.manifold import TSNE

from dataparser import DataParser

class Dimensionality_Reduction(DataParser):
    
    def __init__(self, mat_path=None, data_parser=None):
        """

        Parameters
        ----------
        mat_path : String, optional
            The path to the .mat file
        
        data_parser : String, optional
            DataParser object containing references to data, mask, mzs

        """
        
        if data_parser is not None:
            self.data_parser = data_parser
            self.data = data_parser.data
            self.mzs = data_parser.mzs
            self.mask = data_parser.mask
            self.name = data_parser.name
        elif mat_path is not None:
            super().__init__(mat_path)
        else:
            raise ValueError("A DataParser instance or a .mat file path must be provided.")

    def tSNE_creation(self, n_components=3, perplexity=30, exaggeration=12, learning_rate='auto', init="random", metric="cosine"):
        """

        Parameters
        ----------
        Parameters for t-SNE dimensionality reduction can be found here: https://scikit-learn.org/1.4/modules/generated/sklearn.manifold.TSNE.html
        and are the same parameters used in this function

        Returns
        -------
        X_transformed : numpy array
            The reduced embedding
        emb_dict : Dict
            Dictionary of the embedding and parameters used in the embedding process

        """
            
        if init == 'default':
            init='random'
        start = time.time()
        tSNE_3D = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=exaggeration, learning_rate=learning_rate, init=init, metric=metric, random_state=None)
        TCs_3D = tSNE_3D.fit_transform(self.data)
        end = time.time()
        print(end-start, "Time elapsed")
        X_transformed = TCs_3D
        print(X_transformed.shape, "tsne shape")
        
        emb_dict = {}
        emb_dict["name"] = "tSNE_"+str(init)+str(perplexity)+"_p_"+str(exaggeration)+"_e_"
        emb_dict["n_components"] = n_components
        emb_dict["init"] = init
        emb_dict["distance"] = metric
        emb_dict["perplexity"] = perplexity
        emb_dict["exaggeration"] = exaggeration
        emb_dict["learning_rate"] = learning_rate
        emb_dict["embedded_data"] = X_transformed
        emb_dict["mask"] = self.mask
        
        return X_transformed, emb_dict