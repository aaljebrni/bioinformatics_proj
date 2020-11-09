import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

   
    


   
    
def visualize_Metrics(var, name, learning_rate):
    
    plt.plot(np.squeeze(var))
    plt.ylabel(name)
    plt.xlabel('Folds')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    
    
    
def heatmap(cm, title):
    
    #term2id=eval(open('/content/drive/My Drive/Gene_Chip_Data/bioinformatics_project_data/Characteristics_DiseaseState_classes_dict.txt', 'r').read())
    #id2term = {value : key for (key, value) in term2id.items()}
    ax = plt.axes()
    sns.heatmap(cm, ax = ax, cmap='mako_r')

    ax.set_title(title)
    plt.show()
    

def Visualize_train_test(train, test, name):
       
    plt.ion()
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    plt.plot(train,label= ' Training '+name)
    plt.plot( test ,color='red',label='Testing '+name)
    ax1.set_title("Training and Testing "+name)
#    line1, = ax1.plot(test[1])
#    line2, = ax1.plot(ts_p * 0.6)
   
    plt.legend(loc='upper right')
#    plt.ylim(-1, 2.0)
    plt.show()
    
    
def Visualize_folds(folds, name, flag,y):
       
    plt.ion()
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    plt.ylabel(y)
    plt.xlabel('Epochs')
    plt.plot(folds[0],label= ' 1st Fold')
    plt.plot( folds[1] ,label='2nd Fold ')
    plt.plot(folds[2],label= ' 3rd Fold ')
    plt.plot( folds[3] ,label='4th Fold ')
    plt.plot(folds[4],label= ' 5th Fold ')
    plt.plot( folds[5] ,label='6th Fold')
    plt.plot(folds[6],label= ' 7th Fold ')
    plt.plot( folds[7] ,label='8th Fold')
    plt.plot( folds[8] ,label='9th Fold')
    plt.plot( folds[9] ,label='10th Fold')
    
    ax1.set_title(name)
#    line1, = ax1.plot(test[1])
#    line2, = ax1.plot(ts_p * 0.6)
    
    
    
    if flag:
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
#    plt.ylim(-1, 2.0)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    