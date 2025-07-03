# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""
import mat73
import numpy as np

class DataParser():
    """Class for parsing .mat data"""
    
    def __init__(self, mat_path=None, emb_dict_path=None, name=None, data=None, mzs=None, mask=None, emb_name=None):

        """

        Parameters
        ----------
        mat_path : String, optional
            The path to the .mat file
        emb_dict_path : String, optional
            The path to the embedding dictionary
        name : String, optional
            Name of the data
        data : Numpy array, optional
            Numpy array
        mzs : Numpy array, optional
            m/z channels
        mask : Bool array, optional
            Bool mask for data
        emb_name : String, optional
            Name of the embedded data

        """
        
        if mat_path is not None:
            self.mat = mat_path
            self.data_dict = self.load_mat()
            self.extract_mat(self.data_dict)
        if emb_dict_path is not None:
            self.emb_dict = self.load_embedding(emb_dict_path)
        if mat_path is None:
            self.name = name
            self.data = data
            self.mzs = mzs
            self.mask = mask
            self.emb_name=emb_name
        
    def save_data(self, path, data_dict):
        np.save(path, data_dict, allow_pickle=True)
    
    def load_mat(self):
        
        try:
            data_dict = mat73.loadmat(self.mat)
            return data_dict
        
        except Exception as e:
            print(f"Error loading .mat file: {e}")
            return {}
    
    def extract_mat(self, data_dict):
        
        try:
            self.name = list(data_dict.keys())[0]
            self.data = data_dict[self.name]["data"]
            self.mzs = data_dict[self.name]["spectralChannels"]
            self.mask = data_dict[self.name]["regionOfInterest"]["pixelSelection"]
            
        except Exception as e:
            print(f"Error extracting from the .mat data dictionary: {e}")
            return {}
        
    def load_embedding(self, emb_dict_path):
        
        emb_dict = np.load(emb_dict_path, allow_pickle=True)
        
        """The following will load all attributes of their respective dimensionality
        reduction dictionary produced from the Dimensionality_Reduction class"""
        
        self.emb_name = emb_dict[()]["name"]
        self.embedded_data = emb_dict[()]["embedded_data"]
        self.n_components = emb_dict[()]["n_components"]
        self.init = emb_dict[()]["init"]
        self.distance = emb_dict[()]["distance"]
        self.mask = emb_dict[()]["mask"]
            
        if "tSNE" in self.emb_name:
            print("tSNE embedding provided")
            self.perplexity = emb_dict[()]["perplexity"]
            self.exaggeration = emb_dict[()]["exaggeration"]
            self.learning_rate = emb_dict[()]["learning_rate"]
            return emb_dict[()]