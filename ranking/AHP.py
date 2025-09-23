import numpy as np

weights = {
    "ROI 1000 mean time": 1,
    "ROI 2000 mean time": 1,
    "ROI 3000 mean time": 1,
    "ROI 4000 mean time": 1,
    "ROI 5000 mean time": 1,
    "Sensor energy reflected": 2,
    "ROI 1000 mean memory": 3,
    "ROI 1000 median memory": 3,
    "ROI 1000 max memory": 3,
    "ROI 2000 mean memory": 3,
    "ROI 2000 median memory": 3,
    "ROI 2000 max memory": 3,
    "ROI 3000 mean memory": 3,
    "ROI 3000 median memory": 3,
    "ROI 3000 max memory": 3,
    "ROI 4000 mean memory": 3,
    "ROI 4000 median memory": 3,
    "ROI 4000 max memory": 3,
    "ROI 5000 mean memory": 3,
    "ROI 5000 median memory": 3,
    "ROI 5000 max memory": 3,
    "Facilidade de implementação": 4,
    "ROI 1000 mean gpu": 5,
    "ROI 1000 median gpu": 5,
    "ROI 1000 max gpu": 5,
    "ROI 2000 mean gpu": 5,
    "ROI 2000 median gpu": 5,
    "ROI 2000 max gpu": 5,
    "ROI 3000 mean gpu": 5,
    "ROI 3000 median gpu": 5,
    "ROI 3000 max gpu": 5,
    "ROI 4000 mean gpu": 5,
    "ROI 4000 median gpu": 5,
    "ROI 4000 max gpu": 5,
    "ROI 5000 mean gpu": 5,
    "ROI 5000 median gpu": 5,
    "ROI 5000 max gpu": 5,
}

criteria = list(weights.keys())
n = len(criteria)

ComparisonMatrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            ComparisonMatrix[i, j] = 1
        else:
            ComparisonMatrix[i, j] = weights[criteria[i]] / weights[criteria[j]]

rank = np.linalg.matrix_rank(ComparisonMatrix)

eigenvalues, eigenvectors = np.linalg.eig(ComparisonMatrix)

principalEigenvalue = np.max(eigenvalues)
p = eigenvectors[:,np.argmax(eigenvalues)]

w = np.abs(p / np.sum(np.abs(p)))

CI = (principalEigenvalue - n)/(n-1)

print(f"Final weights: {w}")
print(f"Consistency Index: {CI}")

np.save(".\AHP_weights", w)

# CI aprox 0 -> RI apox 0, i.e., is acceptable