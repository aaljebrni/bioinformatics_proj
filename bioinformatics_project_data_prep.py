import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow.keras.layers as layers 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

from tqdm import tqdm
from imblearn.over_sampling import SMOTE



def standard_scale(x_train, x_test): 
    
    preprocessor = prep.MinMaxScaler().fit(x_train) 
    x_train = preprocessor.transform(x_train) 
    x_test = preprocessor.transform(x_test)
    return x_train, x_test 
  
def get_random_block_from_data(data, batch_size): 
    start_index = np.random.randint(0, len(data) - batch_size) 
    return data[start_index:(start_index + batch_size)] 


# this is the data after reducing its dimensionality using either the DAE I created or the implementation of sklearn for PCA and CPCA
data =  pd.read_csv('pca_reduced_ds_256.csv').astype('float32')


data = data.values
perm = np.random.permutation(data.shape[0])
data = data[perm,:]
X = data[:, :data.shape[1]-1]
Y = data[:,data.shape[1]-1]


# pca = KernelPCA(n_components=500)
# scaler = prep.MinMaxScaler()

# scaler.fit(X)

# X = scaler.transform(X)

# pca.fit(X)
# X = pca.transform(X)
