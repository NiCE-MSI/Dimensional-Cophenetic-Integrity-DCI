# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""

import os
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer
import matplotlib.pyplot as plt

from dimensionalityreduction import Dimensionality_Reduction
from dci import DCI

class DCIopt():
    
    def __init__(self, dataparser, savepath, kwargs):
        """

        Parameters
        ----------
        dataparser : DataParser object, optional
            a dataparser object
        savepath : String, optional
            a savepath for output dictionaries
        kwargs : Dict, optional
            A dictionary of dimensionality reduction parameters, dci parameters, and bayesion optimisation parameters

        """
        self.dataparser = dataparser
        self.savepath = savepath
        self.kwargs = kwargs
        self.algo = kwargs['algo']
        self.neighbours = kwargs['neighbours']
        self.exaggeration = kwargs['exaggeration']
        self.iters = kwargs['iters']
        self.init = kwargs['init']
        self.sampling = kwargs['sampling']
        self.percent = kwargs['percent']
        self.num_pixels = kwargs['num_pixels']
        
    def runopt(self):
        """

        Parameters
        ----------
        algo : String
            The dimensionality reduction algorithm
        neighbours : Int
            Number of neighbours to use in the embedding process, for t-SNE this is perplexity
        exaggeration : Int
            The t-SNE hyperparameter exaggeration
        iters : Int
            How many iterations the Bayesion optimisation should do. This parameter is 10 minimum.

        Returns
        -------
        score_list : List
            A list of all DCI scores across optimisation iterations
        best_params : Float
            The best parameters across the optimisation iterations in the form perplexity, exaggeration
        """
    
        if self.algo =='tsne':
            search_space = [
                Integer(1, self.neighbours, name='perplexity'),
                Integer(1, self.exaggeration, name='exaggeration')]
            
        self.param_list = []
        self.score_list = []
          
        result = gp_minimize(
            func=self.objective,
            dimensions=search_space,
            n_calls=self.iters,
            n_initial_points=10,
            random_state=42,
            callback=[self.plot_progress],)
        
        plt.plot(-result.func_vals)
        plt.xlabel('Iteration')
        plt.ylabel('Best Score so far')
        plt.title('Convergence of Bayesian Optimisation')
        plt.show()
         
        # Best parameters and score
        best_score = -result.fun
        self.best_params = result.x
        print("Best score achieved: ", best_score)
        print("Best parameters found: ", self.best_params)
        
        return self.score_list, self.best_params
        
    def objective(self, params):
        
        """
            Function to define the objective criterion.
            
            :param algo: The dimensionality reduction algorithm
            :type algo: string
            :param dataparser: a dataparser object
            :type dataparser: DataParser object
            :param sampling: Either 'cluster' or 'random' and is the sampling method for DCI
            :type sampling: 
            :param num_pixels: Number of random pixels used for the 'random' sampling method
            :type num_pixels: int
            :param percent: The percentage of total data samples to be used for the 'cluster' sampling method.
            :type percent: int
            :return: a list of scores and the highest scoring hyperparameters.
            """
        
        if self.algo == 'tsne':
            neighbours, exaggeration= params
            dr = Dimensionality_Reduction(data_parser=self.dataparser)
            embedding, emb_dict = dr.tSNE_creation(perplexity=neighbours, exaggeration=exaggeration, init=self.init)
        
        self.dataparser.emb_name = emb_dict['name']
        self.dataparser.embedded_data=emb_dict['embedded_data']
        
        save_path = self.savepath+"\\"+emb_dict['name']
        self.dataparser.save_data(save_path,emb_dict)

        dci_obj = DCI(embedding, data_parser=self.dataparser)
        score, dci_dict, emb_img, fig = dci_obj.dci(sampling=self.sampling, num_pixels=self.num_pixels, percentage=self.percent)
        save_path = self.savepath+"\\"+emb_dict['name']+" DCI DICT"
        self.dataparser.save_data(save_path,dci_dict)
        
        self.score_list.append(score)
        self.param_list.append(params)
        return -score
    
    def moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    def plot_progress(self, res):
        print(self.param_list, "Param list")
        print(self.score_list, "Score list")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(-res.func_vals, marker='o', linestyle='-', color='b')
        ax1.set_title('Current Iteration Results')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Best Score so far')
        smoothed_scores = self.moving_average(-res.func_vals)
        ax2.plot(smoothed_scores, color='r', label="Smoothed Best Score")
        ax2.set_title('Smoothed Convergence Plot')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Best Score (Smoothed)')
        ax2.legend()
        plt.tight_layout()
        temp_save_path = self.savepath+"\\"+"Optimisation plot -Live updates.png"
        final_save_path = self.savepath+"\\"+"Optimisation plot.png"
        fig.savefig(temp_save_path)
        try:
            os.replace(temp_save_path, final_save_path)
        except PermissionError:
            pass
        plt.show()