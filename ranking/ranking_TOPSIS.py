import pandas as pd
import numpy as np

results = pd.read_csv('summary_gpu_memory_times.tsv', sep='\t')
results_array = results.to_numpy()[:, 1:].astype(np.float64)

norms = np.linalg.norm(results_array, ord=2, axis=0)

R = results_array/norms

weights = np.load("AHP_weights.npy")

V = weights * R

A_star = np.min(V, axis=0) # ideal solution A*
A_star[-1] = np.max(V[:, -1])

A_prime = np.max(V, axis=0) # negative ideal solution A’
A_prime[-1] = np.min(V[:, -1])

S_star = np.sqrt(np.sum((A_star - V)**2, axis=1)) # separation from ideal
S_prime = np.sqrt(np.sum((A_prime - V)**2, axis=1)) # separation from negative ideal

C_star = S_prime / (S_prime + S_star) # relative closeness to the ideal solution

ranking = pd.DataFrame({
    "Criterion": results.iloc[:, 0],
    "C_star": C_star
})

ranking = ranking.sort_values(by="C_star", ascending=False).reset_index(drop=True)

print(ranking)