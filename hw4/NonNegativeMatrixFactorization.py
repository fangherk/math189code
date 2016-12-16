
# coding: utf-8

# We consider the dataset found at  
# http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html

# In[1]:

# import basic libraries
# using the nltk.corpus because the data is already
# formatted in an easy to use manner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import reuters
from sklearn.feature_extraction import text
get_ipython().magic('matplotlib inline')


# In[14]:

# gather the complete data matrix
V = np.array([" ".join(list(reuters.words(file_id))).lower() for file_id in reuters.fileids()])
# use the tf-idf extraction method from the sklearn to get the tf-idf features from
# raw text
pretty = text.TfidfVectorizer()
# Learn vocabulary and idf, return term-document matrix.
V = pretty.fit_transform(V)


# In[2]:

# define a simple cost function for the frobenius norm
def cost_function(V, W, H):
    """ general cost function 
    
    V - complete data matrix
    W - left hand side of matrix
    H - right hand side of matrix
    """
    cost = 0
    # return the coordinate matrix of some compelte data matrix
    modded = V.tocoo() 
    # add each row * column set
    for row, col, v in zip(modded.row, modded.col, modded.data):
         cost += np.square(v - np.inner(W[row], H[:,col]))
    return cost


# In[30]:

def nonNegFac(V, r=20, iters=100, eps = 1e-4):
    """ Non-negative matrix factorization on complete data matrix.
    
    V - complete data matrix
    r - reduced parameter size of the matrix
    """
    
    # generate the smaller matrices from V by taking
    # random values, multiply to shrink the total size
    # and make convergence easier
    W = np.abs(np.random.randn(V.shape[0], r) * 1e-3)
    H = np.abs(np.random.randn(r, V.shape[1]) * 1e-3)
    
    # add the first cost function
    costs = [cost_function(V, W, H)]
    diff = costs[0]
    # iterate through given set of iters 
    # copy the paper's implementation 
    i = 0
    
    # use a tolerance, so we do not need to waste time
    while diff > eps and i < iters:
        if i % 2 == 0:
            print("Iteration:{} \t Cost:{}".format(i, costs[-1]))
        H = H*(W.T @ V) / ((W.T @ W) @ H)
        W = W*(V @ H.T) / (W @ (H @ H.T))
        
        #add the cost again
        costs.append(cost_function(V, W, H))
        diff = abs(costs[-1] - costs[-2])
        i += 1
            
    return W, H.T, costs


# In[31]:

W, H, costs = nonNegFac(V, 20, 100)
plt.plot(costs, "r")
plt.title("Non Negative Matrix Factorization")
plt.xlabel("iter")
plt.ylabel("cost")


# In[53]:

top = np.array(pretty.get_feature_names()) # turn the feature names into an array
# essentially we have a list of features now
top = top[np.argsort(H, axis=0)]# get by row
top = top[::-1]
for i in top[:20]:
    print(i)


# In[ ]:



