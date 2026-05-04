import pandas as pd
import numpy as np

results = pd.read_csv('summary_gpu_memory_times.tsv', sep='\t')
results_array = results.to_numpy()[:, 1:].astype(np.float64)

weights = np.load("AHP_weights.npy")

# normalization in [0,1] with inversion
R = 1 - (results_array - np.min(results_array, axis=0))/(np.max(results_array, axis=0) - np.min(results_array, axis=0))
R[:, -1] = (results_array[:, -1] - np.min(results_array[:, -1]))/(np.max(results_array[:, -1]) - np.min(results_array[:, -1]))

V = weights * R

A_star = np.min(V, axis=0) # ideal solution A*
