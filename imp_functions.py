
# coding: utf-8

# In[1]:

import numpy as np

def split_data(x, y, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


# In[2]:


def predict(x,w):
    y_pred =  np.dot(x,w)
    y_pred[y_pred>0.01] = 1
    y_pred[y_pred<=0.01] = -1 
    return y_pred


# In[3]:


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly,np.power(x, deg)]
    return poly


# In[6]:


def remove_outliers(mat):
    ''' replace outlier (-999) with median of the feature'''
    cols_to_remove = []              #remove the column if entire column contain outliers
    for i in range(0,mat.shape[1]):
        col = [x for x in mat[:,i] if x!=-999]    #list contain only inliners
        if len(col) == 0:                         # if no inliners 
            cols_to_remove.append(i)
        else :
            median = np.median(col)
            index = np.where(mat[:,i] == -999)
            mat[index,i] = median                #replace outliers with median
    return mat, cols_to_remove


# In[7]:


def normalize(mat, is_test_data=0, train_mean=None, train_std=None):
    if is_test_data:                        #if test data then normalize it with the mean and std of training data
        mat = (mat-train_mean)/train_std              
        return mat
        
    mean = np.mean(mat,axis=0)          # for training data compute mean and std and normalize the data
    std = np.std(mat,axis=0)
    mat = (mat-mean)/std
    return mat, mean, std


# In[ ]:


def remove_high_correlation(data, threshold, visualisation=False):

    corr = np.corrcoef(data, rowvar=False)
    high_corr = corr > threshold

    diagonal = np.arange(high_corr.shape[0])
    high_corr[diagonal, diagonal] = False
    high_corr[np.tril_indices(high_corr.shape[0])] = False

    correlated = np.nonzero(high_corr)

    data = np.array(np.delete(data, np.unique(correlated[1]), axis=1))
    print('Removing columns: ', np.unique(correlated[1]))
    
    
    if visualisation:
        plt.figure(figsize=(10, 20))

        plt.subplot(1,2,1)
        sns.heatmap(corr)

        plt.subplot(1,2,2)
        sns.heatmap(high_corr)
        
    return data, np.unique(correlated[1])


# In[8]:


def remove_low_correlation_with_label(data,y, threshold):
    
    data = np.c_[data,y]
    corr = np.corrcoef(data, rowvar=False)
    corr = corr[:-1,-1]
    
    low_corr = np.nonzero(abs(corr) < threshold)

    data = np.array(np.delete(data, low_corr[0], axis=1))
    print('Removing columns: ', (low_corr[0]))

    return data[:,:-1], low_corr[0]


# In[9]:


def build_model_data(features):
    """Form (y,tX) to get regression data in matrix form."""
    x = features
    num_samples = features.shape[0]
    tx = np.c_[x,np.ones(num_samples)]
    return tx


# In[10]:


def convert_label(y,check):
    ''' for logistic regression convert labels from -1 to zero for NLL loss function'''
    if check == -1:
        y[y<=0] = -1   # for predcition convert back to -1
        return y
    y[y<0] = 0      # convert -1 to 0 for NLL loss
    return y


# In[11]:


def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]

