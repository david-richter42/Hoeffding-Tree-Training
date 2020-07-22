""" Training a Hoeffding Tree on numpy arrays:

X shape: (len(X) // 3, 1, 3)
Each sample contains three values, which is why the x value is divided by 3. 
Example X sample: np.array([[0.5, 0.5, 0.5]])

y shape: (len(X) // 3, 1)
For the targets, it has to be a numpy array of single-element integer numpy arrays.
Example y sample: np.array([0.5])
"""

import os
import numpy as np

from skmultiflow.trees import HoeffdingTreeClassifier

num_samples = 30000
samples = np.random.rand(num_samples)
samples = samples.reshape(len(samples) // 3, 1, 3)
targets = np.random.randint(2, size=len(samples))
targets = targets.reshape(len(samples), 1)

ht = HoeffdingTreeClassifier()

correct = 0
max_samples = num_samples // 3
for i in range(max_samples):
    X, y = samples[i], targets[i]
    y_pred = ht.predict(X)
    if y[0] == y_pred[0]:
        correct += 1
    ht = ht.partial_fit(X, y)

    # This is just here so I knew it was working
    # print(f"processed sample: {i}")

print(f"Hoeffding Tree accuracy: {correct / len(samples)}")