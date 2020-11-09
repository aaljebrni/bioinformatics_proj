import pandas as pd
import numpy as np
import tensorflow as tf
import lightgbm as lgb

from tensorflow.keras import Model
from tensorflow.keras import layers
from bioinformatics_project_data_prep import *
from bioinformatics_project_data_vis import *
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold,StratifiedKFold, StratifiedShuffleSplit
from sklearn import metrics
from imblearn.over_sampling import SMOTE


class Diseases_classifier(Model): 
    
  
    def __init__(self, 
                loss_object,
                optimizer,
                train_loss,
                train_metric,
                test_loss,
                dims,
                test_metric): 
        super(Diseases_classifier, self).__init__() 
        
        
        self.h_layer1 = layers.Dense(dims[0], activation ='relu', kernel_regularizer=regularizers.l2(0.002)) 
        self.h_layer2 = layers.Dense(dims[1], activation ='relu', kernel_regularizer=regularizers.l2(0.002)) 
        self.h_layer3 = layers.Dense(dims[2], activation ='relu', kernel_regularizer=regularizers.l2(0.002)) 
        self.h_layer4 = layers.Dense(dims[3], activation ='relu', kernel_regularizer=regularizers.l2(0.002)) 
        self.out_layer = layers.Dense(dims[4], activation = 'softmax')
        self.drp = layers.Dropout(0.5)
        
        self.dims = dims 
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_metric = train_metric
        self.test_loss = test_loss
        self.test_metric = test_metric
        
        self.train_loss_per_epoch = []
        self.train_accuracy_per_epoch = []

        self.validation_loss_per_epoch = []
        self.validation_accuracy_per_epoch = []
        
        
        
        
    # @tf.function         
    def model(self, inputs): 
        l1 = self.h_layer1(inputs) 
        l2 = self.h_layer2(l1) 
        l3 = self.h_layer3(l2)
        l4 = self.h_layer4(l3)
        
        return self.out_layer(l4) 
    
    @tf.function
    def train_step(self, features, labels):
        
        with tf.GradientTape() as tape:
            predictions = self.model(features)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        
        self.train_loss(loss)
        self.train_metric(labels, predictions)
        
    @tf.function
    def test_step(self, features, labels):
        
        predictions = self.model(features)
        t_loss = self.loss_object(labels, predictions)
        
        self.test_loss(t_loss)
        self.test_metric(labels, predictions)
        
        
    def fit(self, train, test, fold, epochs):
        
        
        
        for epoch in range(epochs):
            for features, labels in train:
                self.train_step(features, labels)
            
            
            
            
            template = 'Fold {0}, Epoch {1}, Loss: {2}, Accuracy: {3}'
            print(template.format(fold, epoch+1,
                                  self.train_loss.result(),
                                  self.train_metric.result()*100))
                                  
            
            self.train_loss_per_epoch.append(self.train_loss.result().numpy())
            self.train_accuracy_per_epoch.append( self.train_metric.result().numpy()*100)

            
            #Reset the metrics for the next epochs
            self.train_loss.reset_states()
            self.train_metric.reset_states()
            














# 10 fold cross-validation
train_loss_per_fold = []
train_accuracy_per_fold = []
    
valid_loss_per_fold = []
valid_accuracy_per_fold = []
    
epochs_train_loss_per_fold = []
epochs_train_accuracy_per_fold = []
    
epochs_valid_loss_per_fold = []
epochs_valid_accuracy_per_fold = []
    
train_cm_per_fold = []
valid_cm_per_fold = []

train_report_per_fold = []
valid_report_per_fold = []
    
n_splits = 3
i=1


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)


for train_index, test_index in sss.split(X, Y):
        x_train,x_test=X[train_index],X[test_index]
        y_train,y_test=Y[train_index],Y[test_index]
        
      
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    
    
        # choose an optimizer
        learning_rate = 0.0001
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    
        #specify metrics for training 
        train_loss = tf.keras.metrics.Mean(name='tain_loss')
        train_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    
        #specify metrics for testing
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
        dims =[1024,512, 256, 128, 51]
        # dims = [1392, 2785, 5570, 22283, 52], [1392, 2785, 5570, 10000, 52], [1000, 750, 500, 250, 52]
        #create an instance of the model
    
        model = Diseases_classifier(loss_object = loss_object, optimizer = optimizer, train_loss = train_loss, train_metric = train_metric, test_loss = test_loss, test_metric = test_metric, dims = dims)
        EPOCHS = 100
    
    
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(buffer_size = x_train.shape[0]) 
        
    
        model.fit(train = train_data, test = None, fold= i, epochs= EPOCHS)
    
       
        train_preds = model.model(x_train)
        train_accuracy = model.train_metric(y_train, train_preds).numpy()
        train_loss = model.loss_object(y_train, train_preds).numpy()
        train_cm = tf.math.confusion_matrix(y_train, train_preds.numpy().argmax(axis=1), num_classes=51)
        
        
        #populating the training lists per fold
        train_loss_per_fold.append(np.mean(model.train_loss_per_epoch))
        train_accuracy_per_fold.append(np.mean(model.train_accuracy_per_epoch))
        train_cm_per_fold.append(train_cm)
        train_report_per_fold.append( metrics.classification_report(y_train, train_preds.numpy().argmax(axis=1), digits=3, zero_division=True, output_dict=True) )
        #populating train epochs per fold lists
        epochs_train_loss_per_fold.append(model.train_loss_per_epoch)
        epochs_train_accuracy_per_fold.append(model.train_accuracy_per_epoch)
        
        
        
        
        
        
        test_preds = model.model(x_test)
        test_accuracy = model.test_metric(y_test, test_preds).numpy()
        test_loss = model.loss_object(y_test, test_preds).numpy()
        test_cm = tf.math.confusion_matrix(y_test, test_preds.numpy().argmax(axis=1), num_classes=51)
        
        
        
        #populating the validation lists per fold
        valid_loss_per_fold.append(test_loss)
        valid_accuracy_per_fold.append(test_accuracy*100)
        valid_cm_per_fold.append(test_cm)
        valid_report_per_fold.append(metrics.classification_report(y_test, test_preds.numpy().argmax(axis=1), digits=3, zero_division=True, output_dict=True) )
        print('Fold: {0} Validation loss: {1}, Validation Accuracy: {2}'.format(i, test_loss, test_accuracy*100))
        # heatmap(test_cm, 'Testing Confusion Matrix')
        i += 1
        
        
        
#Visualizing the results

mean_valid_cm=np.mean(valid_cm_per_fold, axis=0)
sns.heatmap(mean_valid_cm, cmap='winter_r', xticklabels=50,yticklabels=50)
Visualize_folds(epochs_train_loss_per_fold, "Training loss per fold",False,"Loss")
Visualize_folds(epochs_train_loss_per_fold, "Training accuracy per fold",True,"Accuracy")