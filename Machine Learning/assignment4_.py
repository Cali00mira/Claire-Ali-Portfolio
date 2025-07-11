import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

mu = 0
var = 1
n = (10000,1)

# latent features
f1 = np.random.normal(loc=mu,scale=var,size=n)
f2 = np.random.normal(loc=mu,scale=var,size=n)

# observed features
e = np.random.normal(loc=mu,scale=var,size=n)
x1 = 2*f1 + 3*f2 + e
x2 =f1 - 10*f2 + e
x3 = -5*f1 + 5*f2 + e
x4 = 7*f1 + 11*f2 + e
x5 = -6*f1 - 7*f2 + e

# scaling 
ss = StandardScaler()
f1 = pd.DataFrame(ss.fit_transform(f1))
f2 = pd.DataFrame(ss.fit_transform(f2))
X = pd.DataFrame(np.hstack((x1,x2,x3,x4,x5)))
X = pd.DataFrame(ss.fit_transform(X))


# reduce via PCA and plot
X_pca = PCA(n_components=2).fit_transform(X)

fig1 , axs1 = plt.subplots(2,2)

axs1[0][0].scatter(f1,X_pca[:,0])
axs1[0, 0].set_title('f1 vs component 1')

axs1[1][0].scatter(f1,X_pca[:,1])
axs1[1, 0].set_title('f1 vs component 2')

axs1[0][1].scatter(f2,X_pca[:,0])
axs1[0, 1].set_title('f2 vs component 1')

axs1[1][1].scatter(f2,X_pca[:,1])
axs1[1,1].set_title('f2 vs component 2')

fig1.suptitle('Subplots for PCA', fontsize=16)
fig1.tight_layout()

# reduce with factor analysis and plot
X_fa = FactorAnalysis(n_components=2,max_iter=10000).fit_transform(X)

fig2 , axs2 = plt.subplots(2,2)

axs2[0][0].scatter(f1,X_fa[:,0])
axs2[0, 0].set_title('f1 vs component 1')

axs2[1][0].scatter(f1,X_fa[:,1])
axs2[1, 0].set_title('f1 vs component 2')

axs2[0][1].scatter(f2,X_fa[:,0])
axs2[0, 1].set_title('f2 vs component 1')

axs2[1][1].scatter(f2,X_fa[:,1])
axs2[1,1].set_title('f2 vs component 2')

fig2.suptitle('Subplots for Factor Analysis', fontsize=16)
fig2.tight_layout()
plt.show()

