import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import itertools
from sklearn.metrics.cluster import adjusted_rand_score
import plotly.express as px


# generate dataset of random clusters
x1 = np.random.multivariate_normal([1, 2], [[0.1, 0.05], [0.05, 0.2]], 1000)
x2 = np.random.multivariate_normal([0, 0], [[0.3, -0.1], [-0.1, 0.2]], 500)
x3 = np.random.multivariate_normal([-2, 3], [[1.5, 0], [0, 1.5]], 1500)
X = np.concatenate((x1,x2,x3))

# set ground truth values
g_true = np.concatenate((np.zeros(len(x1)), np.ones(len(x2)), 2 * np.ones(len(x3))))

# parameter tuning
e = np.arange(0.1,1.1,0.1)
ms = np.arange(1,11,1)
combo = list(itertools.product(e,ms))

def test(p,q):
    db = DBSCAN(eps=p, min_samples=q).fit(X)
    labels = db.labels_
    ad = adjusted_rand_score(g_true,labels)
    return ad

# iterate to find scores for combinations of parameters
lis = [test(*c) for c in combo]

# get index and parameters of highest score
idx = lis.index(max(lis))
e1 = combo[idx][0]
ms1 = combo[idx][1]

print(f"the highest adjusted Rand index value is: ",{max(lis)})
print(f"the best eps value is: ",{e1})
print(f"the best min_samples value is: ",{ms1})

# fit cluster and plot
X = pd.DataFrame(X, columns=['mean radius','mean fractal dimension'])
db = DBSCAN(eps=e1,min_samples=ms1).fit(X)
X['cluster'] = db.labels_.astype(str)

# needs to be run in interactive window
px.scatter(X, x="mean radius", y="mean fractal dimension", color='cluster') 