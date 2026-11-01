# -*- coding: utf-8 -*-
"""
@author: cn9
"""
import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist
from dataparser import DataParser
from plotting import plots

def main(matpath1, matpath2):
    dataparser1 = DataParser(mat_path = matpath1)
    random_pixels1 = list(range(len(dataparser1.data)))
    raw_pixels1 = [dataparser1.data[i,:] for i in random_pixels1]
    raw_distances1 = pdist(raw_pixels1, metric="cosine")
    data_plot1 = plots(raw_distances1, labels=random_pixels1)
    data1_cophenetic_distance, data1_cophenetic_matrix, linkage_matrix1 = data_plot1.agglomerative_clustering_dendrogram(metric="cosine")
   
    dataparser2 = DataParser(mat_path = matpath2)
    random_pixels2 = list(range(len(dataparser2.data)))
    raw_pixels2 = [dataparser2.data[i,:] for i in random_pixels2]
    raw_distances2 = pdist(raw_pixels2, metric="cosine")
    data_plot2 = plots(raw_distances2, labels=random_pixels2)
    data2_cophenetic_distance, data2_cophenetic_matrix, linkage_matrix2 = data_plot2.agglomerative_clustering_dendrogram(metric="cosine")
   
    emd = compare_coph_emd(data1_cophenetic_matrix, data2_cophenetic_matrix)
    return emd

def d_sort(coph_matrix):
    sorted_coph_matrix = squareform(coph_matrix)
    d1_normalized = (sorted_coph_matrix-np.min(sorted_coph_matrix)) / (np.max(sorted_coph_matrix) - np.min(sorted_coph_matrix))
    d1_sorted = np.sort(d1_normalized)
    cdf1 = np.arange(1, len(sorted_coph_matrix)+1)
    cdf1 = cdf1/len(sorted_coph_matrix)
    return d1_sorted, cdf1
   
def calc_freq(sorted_arr, title="Cophenetic distance ranges vs frequency of occurrence"):
    division = [i / 100 for i in range(0, 102, 1)]
    frequencies, _ = np.histogram(sorted_arr, bins=division)
    frequencies = frequencies/(len(sorted_arr))
    return frequencies, division

def do_emd(f1, d1, f2, d2):
    emd = wasserstein_distance(d1[:-1], d2[:-1], f1, f2)
    return emd
   
def compare_coph_emd(data1_cophenetic_matrix, data2_cophenetic_matrix):
    d1_sorted, cdf1 = d_sort(data1_cophenetic_matrix)
    f1, d1 = calc_freq(d1_sorted)
   
    d2_sorted, cdf2 = d_sort(data2_cophenetic_matrix)
    f2, d2 = calc_freq(d2_sorted)
   
    emd = do_emd(f1, d1, f2, d2)
    return emd