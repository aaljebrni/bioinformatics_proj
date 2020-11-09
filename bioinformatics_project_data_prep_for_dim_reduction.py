import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow.keras.layers as layers 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE



def standard_scale(X_train, X_test): 
    preprocessor = prep.MinMaxScaler().fit(X_train) 
    X_train = preprocessor.transform(X_train) 
    X_test = preprocessor.transform(X_test) 
    return X_train, X_test 
  
def get_random_block_from_data(data, batch_size): 
    start_index = np.random.randint(0, len(data) - batch_size) 
    return data[start_index:(start_index + batch_size)] 

def scale(X): 
    preprocessor = prep.MinMaxScaler().fit(X) 
    X = preprocessor.transform(X) 
    
    return X

data =  pd.read_csv('/content/drive/My Drive/Gene_Chip_Data/bioinformatics_project_data/'+'Characteristics_DiseaseState_ds_51.csv').astype('float32')






data = data.values
perm = np.random.permutation(data.shape[0])
data = data[perm,:]
X = data[:, :data.shape[1]-1]
Y = data[:,data.shape[1]-1]


#********************************in case u wanna SMOTE the whole ds for the DAE******************************************
# preprocessor = prep.MinMaxScaler().fit(X)
# X = preprocessor.transform(X) 

# smote = SMOTE(sampling_strategy='not majority')
# X,Y= smote.fit_sample(X,Y)
#***************************************************************************
x_train, x_test, y_train, y_test =train_test_split(X, Y,   test_size=0.2) 

x_train, x_test = standard_scale(x_train, x_test)


smote = SMOTE(sampling_strategy='not majority')
x_train,y_train= smote.fit_sample(x_train,y_train)

print('Data is ready!')
