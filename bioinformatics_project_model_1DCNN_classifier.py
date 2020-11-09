import tensorflow as tf
from tensorflow.keras import regularizers

from bioinformatics_project_data_prep import *
from bioinformatics_project_data_vis import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold,StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from tensorflow.keras import regularizers






def cnn_classifier(x_train, y_train, x_test, y_test):
    
    
    
   
    model =tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu',kernel_regularizer=regularizers.l2(0.002), padding = 'same', input_shape=x_train.shape[1:3]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.002), padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(0.002), padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(51, activation='softmax'))
    
    
    
    
    return model
    

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
    
n_splits = 5
i=1


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)


for train_index, test_index in sss.split(X, Y):
    
    x_train,x_test=X[train_index],X[test_index]
    y_train,y_test=Y[train_index],Y[test_index]
    
    
    print('training set shape before sampling: ', x_train.shape)
    
    
    print('training set shape after sampling: ', x_train.shape)
    
    verbose, epochs, batch_size = 1, 100, 32
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = cnn_classifier(x_train, y_train, x_test, y_test)
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['sparse_categorical_accuracy'])
    print('Fold No: ', i)
    history = model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = verbose)
    
    epochs_train_loss_per_fold.append(history.history['loss'])
    epochs_train_accuracy_per_fold.append(np.asarray(history.history['sparse_categorical_accuracy']) * 100)
    
    valid_loss, valid_accuracy = model.evaluate(x_test, y_test, batch_size = batch_size, verbose = 0)
    valid_preds = model.predict(x_test)
    valid_cm = tf.math.confusion_matrix(y_test, valid_preds.argmax(axis=1), num_classes=51)
    valid_cm_per_fold.append(valid_cm)
    valid_loss_per_fold.append(valid_loss)
    valid_accuracy_per_fold.append(valid_accuracy*100)
    valid_report_per_fold.append(metrics.classification_report(y_test, valid_preds.argmax(axis=1), digits=3, zero_division=True, output_dict=True) )
    
    train_loss, train_accuracy = model.evaluate(x_train, y_train, batch_size = batch_size, verbose = 0)
    train_preds = model.predict(x_train)
    train_cm = tf.math.confusion_matrix(y_train, train_preds.argmax(axis=1), num_classes=51)
    train_cm_per_fold.append(train_cm)
    train_loss_per_fold.append(train_loss)
    train_accuracy_per_fold.append(train_accuracy*100)
    train_report_per_fold.append( metrics.classification_report(y_train, train_preds.argmax(axis=1), digits=3, zero_division=True, output_dict=True) )
    print('Fold: {0} Validation loss: {1}, Validation Accuracy: {2}'.format(i, valid_loss, valid_accuracy*100))
    i += 1
    
   
    
#Visualizing the results

mean_valid_cm=np.mean(valid_cm_per_fold, axis=0)
sns.heatmap(mean_valid_cm, cmap='winter_r', xticklabels=50,yticklabels=50)
Visualize_folds(epochs_train_loss_per_fold, "Training loss per fold",False,"Loss")
Visualize_folds(epochs_train_loss_per_fold, "Training accuracy per fold",True,"Accuracy")