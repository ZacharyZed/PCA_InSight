# EDA
# PCA analysis for UF Membrane
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 08:34:29 2017

@author: xxxxxx685
"""

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pylab import rcParams
import seaborn as sns

uf = pd.read_excel('artier.xlsx', sheetname = 'UF2')

uf = uf.fillna(method='ffill')
uf = uf.fillna(uf.mean())

uf = uf.drop(['date'], axis = 1)
ss = StandardScaler() 
col = (uf.columns.values) 

uf1_v2 = ss.fit_transform(uf) # scale 
uf = pd.DataFrame(uf1_v2, columns = col)

correlation_matrix = uf.corr() # correlation matrix
corrm = correlation_matrix.describe()
#correlation_matrix.to_excel('sheet.xlsx')

n = len(col) # component length
pca = PCA(n_components = n) 
pca.fit(uf)

x = pca.fit_transform(uf)

#scree

num_vars = n
num_obs = 9
A = np.random.randn(num_obs, num_vars)
A = np.asmatrix(A.T) * np.asmatrix(A)
U, S, V = np.linalg.svd(x) 
eigvals = S**2 / np.cumsum(S)[-1]

fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(num_vars) + 1
plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
leg.draggable(state=True)
plt.show()

##  Setting up visualization for PCA

xvect = pca.components_[0] #PC1
yvect = pca.components_[1] #PC2

xs = pca.transform(uf)[:,0]
ys = pca.transform(uf)[:,1]

# plot PCA
for i in range(len(xvect)):
    plt.arrow(0, 0, xvect[i]*max(xs), yvect[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvect[i]*max(xs)*2.5, yvect[i]*max(ys)*2.5,
             list(uf.columns.values)[i], color='k')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(uf.index)[i], color='k')

for i in range(len(xs)):
    plt.plot(xs[i], ys[i], '^')
    
# plot point from cluster    
clust1 = uf['Avg(VAL) for UF 2.PDR'] # sample var from cluster obs1
clust2 = uf['Avg(VAL) for UF 2.Flux'] # sample var from cluster obs2
clust3 = uf['Avg(VAL) for UF 2.PostBP'] # sample var from cluster obs3

# plot cluster var rep
plt.plot(clust1, 'g-')
plt.plot(clust2, 'b-')
plt.plot(clust3, 'r-')

plt.show()    

#plt.show()

#rcParams['figure.figsize'] = 15, 15
#sns.heatmap(correlation_matrix, vmax =1., square = False).xaxis.tick_top()
