from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import pandas
import matplotlib.pyplot as plt

CLUSTER_SIZE = 4
def solve(Algorithm,title):
    fig = plt.figure()
    fig.suptitle(title)
    data = pandas.read_csv("test.csv").loc[0:501]
    #data = data.drop('CUST_ID',axis=1).fillna(0)
    gm = Algorithm(CLUSTER_SIZE)
    gm.fit(data)
    mds = MDS(n_components=2)
    output = pandas.DataFrame(mds.fit_transform(data),columns=['X','Y'])
    output['clustered'] = gm.predict(data)
    plot = fig.add_subplot()
    
    for i in range(0,CLUSTER_SIZE):
        to_plot = output[output['clustered'] == i]
        plot.scatter(to_plot["X"],to_plot["Y"])

solve(GaussianMixture,"Gaussian")
solve(KMeans,"KMeans")
plt.show()

