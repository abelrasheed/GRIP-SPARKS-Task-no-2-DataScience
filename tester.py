import numpy as np
from kmeans import Kmeans

X = [
    [0.9,0.54],
    [0.865,0.098],
    [0.0009,0.0006]
]

X  = np.array(X)
kmeans = Kmeans(X)
kmeans.train()

