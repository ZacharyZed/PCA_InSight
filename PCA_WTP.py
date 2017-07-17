# EDA
# PCA analysis for UF1 Cartier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pylab import rcParams
import seaborn as sns



# import
uf = pd.read_excel('cartierUF1.xlsx')

print(uf.head(5)) # preview
uf = uf.drop(['fullDate'], axis = 1) # remove date (not of use for PCA)
ss = StandardScaler() # prep StandardScaler
col = (uf.columns.values) # column headers

uf1 = ss.fit_transform(uf) # scale 
uf = pd.DataFrame(uf1, columns = col) # turn to DF

correlation_matrix = uf.corr() # correlation matrix
corrm = correlation_matrix.describe()
print(corrm)

n = len(col) # component length
pca = PCA(n_components = n) 
pca.fit(uf)

x = pca.fit_transform(uf)

# export to desktop - uncomment following lines, replace path with your storage path
# correlation_matrix.to_excel(path)

##  Setting up visualization for PCA

xvect = pca.components_[0] #PC1
yvect = pca.components_[1] #PC2

xs = pca.transform(uf)[:,0]
ys = pca.transform(uf)[:,1]


##  Visualization

# plot PCA
for i in range(len(xvect)):
# arrows represent variables 
    plt.arrow(0, 0, xvect[i]*max(xs), yvect[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvect[i]*max(xs)*2.5, yvect[i]*max(ys)*2.5,
             list(uf.columns.values)[i], color='k')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(uf.index)[i], color='k')

for i in range(len(xs)):
# triangles represent row values in sheet
    plt.plot(xs[i], ys[i], '^')

rcParams['figure.figsize'] = 45, 70 # enlarge plt
plt.show()

rcParams['figure.figsize'] = 15, 5 #downsize plt

clust1 = uf['PressureDecayRate'] # sample var from cluster obs1
clust2 = uf['TCFluxDuringBP'] # sample var from cluster obs2
clust3 = uf['PermeabilityAfterBP'] # sample var from cluster obs3

# plot cluster var rep
plt.plot(clust1, 'g-')
plt.plot(clust2, 'b-')
plt.plot(clust3, 'r-')

plt.show()

##  Seasborn corr heatmap matrix
rcParams['figure.figsize'] = 15, 15 # enlarge plt
sns.heatmap(correlation_matrix, vmax =1., square = False).xaxis.tick_top()



