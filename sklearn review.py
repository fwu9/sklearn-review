import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use('ggplot')

## Q1
## Different from most people, I think the regression coefficient
## cannot represent the influence size of the variable. It is affected by measurement unit.
## I choose to calculate P-value
def linear_model_factor_importance(X,y):
    ''' X is the dataframe of features '''
    ''' y is a nparray of target '''

    lm = LinearRegression()
    lm.fit(X, y)
    params = np.append(lm.intercept_,lm.coef_)
    predictions = lm.predict(X)
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

    var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params/ sd_b


    newX = np.append(np.ones((len(X),1)), X, axis=1)
    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
    p_values = np.round(p_values,10)

    ## Record the results, from greatest to least
    result = pd.DataFrame(p_values[1:].reshape(13,1), index=X.columns, columns=['p_values'])
    print(result.sort_values(by="p_values" , ascending=True))

## Q2
def plot_elbow(X, md = True):

    ''' X is the feature data (arrays) '''
    ''' if md == True, we will use manhanttan distance '''
    ''' else, we use euclidean distance '''

    distance = []
    k = []

    for n_clusters in range(1,10):
       cls = KMeans(n_clusters).fit(X)

       def manhattan_distance(x,y, manhattan_distance = True):
           return np.sum(abs(x-y))

       def euclidean_distance(x,y):
           return np.sum((x-y)**2)

       distance_sum = 0
       for i in range(n_clusters):
          group = cls.labels_ == i
          members = X[group,:]

          if md == True:
              for v in members:
                  distance_sum += manhattan_distance(np.array(v), cls.cluster_centers_[i])
          else:
              for v in members:
                  distance_sum += euclidean_distance(np.array(v), cls.cluster_centers_[i])
       distance.append(distance_sum)
       k.append(n_clusters)
    plt.scatter(k, distance)
    plt.plot(k, distance)
    plt.xlabel("k")
    plt.ylabel("distance")
    plt.show()

if __name__ == '__main__':
    ## Q1
    ## dataset
    boston_dataset = load_boston()
    X = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    y = boston_dataset.target
    linear_model_factor_importance(X, y)

    ## Q2
    iris = datasets.load_iris()
    wine = datasets.load_wine()
    X1 = iris.data
    X2 = wine.data

    plot_elbow(X1, md=True)
    plot_elbow(X2, md=True)
    ## According to the elbow principle, choose n=3 for both datasets