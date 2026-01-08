# -*- coding: utf-8 -*-
"""
@author: Connor James Newstead
"""

from dataparser import DataParser
from dimensionalityreduction import Dimensionality_Reduction
from dci import DCI

#First, create a dataparser object with a local .mat file
dataparser = DataParser(mat_path="synth_data_fixed.mat")

#Then do a dimensionality reduction
tsne_embedding, tsne_dictionary = Dimensionality_Reduction(data_parser=dataparser).tSNE_creation()

#Finally calculate DCI
#Sampling can be 'cluster' or 'random'. If 'random' then 'num_pixels' must be >1
dci, dci_dictionary, emb_img, fig = DCI(tsne_embedding, data_parser=dataparser).dci(sampling='cluster', percentage=1)
