# Dimensional-Cophenetic-Integrity-DCI: A method for evaluating dimensionality reduction
 
> A standalone GUI application implementing DCI evaluation of dimensionality reduction in addition to a Bayesian optimisation method. 
> Data can be found in the Metabolights repository with unique identifier REQ20250618211289
> This package is explained in detail in [Placeholder] and it is shown here for transparency purposes. Please cite this paper if you use this package
 
---
## What This Does
 
**DCI** introduces a method for evaluating the preservation of cluster relationships between high and low dimensional space. The open source code and standalone GUI allows for:
 
- Loading a dataset as a .mat, .npy or .csv
- Apply DCI to process the data or do a Bayesian optimisation using DCI as the objective criterion.
- Visualize the results
- Export output dictionaries
---
 
## Example: How to Process a Dataset
## With GUI

1. Load a .mat saved in v7.3. Within the .mat, data should be saved as 'data', mzs should be saved as 'spectralChannels', and the mask should be saved as 'pixelSelection' within regionOfInterest saved as a struct. Otherwise, the relevant npy/csv files can be loaded.

2. Produce a t-SNE embedding with the desired hyperparameters on the t-SNE page after 'Dimensionality Reduction'.

3. On the 'Open' page, load the embedding dictionary .npy file

4. On the 'DCI' page select the desired sampling method by checking either 'Samples (Cluster method)' or 'Samples (Random Method)' and use the slider or directly change the number beside it. For the 'cluster' method, n% of total data samples are used.

5. Click 'Run DCI' and wait. A .npy dictionary will be output which contains high and low dimensional cophenetic distance matrices, the samples selected, high and low dimensional linkage matrices, centroids (if applicable) and the mutual information between high and low dimensional cophenetic distance matrices.

## Without GUI
See Demo.py to see an example of basic use.
1. Create a DataParser object by loading a .mat saved in v7.3. Within the .mat, data should be saved as 'data', mzs should be saved as 'spectralChannels', and the mask should be saved as 'pixelSelection' within regionOfInterest saved as a struct. Otherwise, the relevant npy/csv files can be loaded. The DataParser class will take mat_path to do this.

2. Using the Dimensionality_Reduction class, produce a t-SNE embedding with the desired hyperparameters under the tSNE_creation() function. Descriptions of what each hyperparameter does can be found on the sklearn t-SNE docs. The transformed reduced representation and a dictionary containing the name of embedding, components, init, distance, perplexity, exaggeration, learning_rate, embedded data and mask is produced.

3. Pass the DCI class the produced embedding and DataParser object, and call the dci function with the optional parameter sampling = 'cluster' or 'random', additionally change the optional parameters percentage or num_pixels to control the number of samples. The output of this function is the mutual information between high and low dimensional cophenetic distance matrices, a DCI results dictionary, an embedding image array (if n_components == 3) and a Matplotlib figure of cophenetic distance matrices.
---
 Download
 
Download the latest release (.exe)
No Python or installation needed.
https://www.dropbox.com/scl/fi/x5mn1x96yhow9atfbj1jb/DCI.exe?rlkey=7y0utajtsf91rbh9npstbj2rm&st=bc3wfl8u&dl=0
---
The standalone GUI does NOT need to be downloaded. This repo can be cloned or downloaded, and the GUI can be run from main.py

Notes
Tested on Windows 10

---
