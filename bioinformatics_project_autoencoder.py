import pandas as pd
import numpy as np
import tensorflow as tf
from  bioinformatics_project_data_prep_for_dim_reduction import *


class Encoder(tf.keras.layers.Layer): 
    
      
    def __init__(self, 
                n_dims, 
                name ='dim_reduction_encoder_class', 
                **kwargs): 
        super(Encoder, self).__init__(name = name, **kwargs) 
        self.n_dims = n_dims 
        self.n_layers = 1
        self.encode_layer1 = layers.Dense(n_dims[0], activation ='relu') 
        self.encode_layer2 = layers.Dense(n_dims[1], activation ='relu') 
        
        self.encode_out = layers.Dense(n_dims[2], activation ='relu') 
          
    @tf.function         
    def call(self, inputs): 
        l1 = self.encode_layer1(inputs)
        l2 = self.encode_layer2(l1)
             
        return self.encode_out(l2) 
  
class Decoder(tf.keras.layers.Layer): 
    
  
    def __init__(self, 
                n_dims, 
                name ='decoder_class', 
                **kwargs): 
        super(Decoder, self).__init__(name = name, **kwargs) 
        self.n_dims = n_dims 
        self.n_layers = len(n_dims) 
        self.decode_layer1 = layers.Dense(n_dims[0], activation ='relu') 
        self.decode_layer2 = layers.Dense(n_dims[1], activation ='relu') 
        
        self.recon_layer = layers.Dense(22283, activation ='sigmoid') 
          
    @tf.function         
    def call(self, inputs): 
        l1 = self.decode_layer1(inputs) 
        l2 = self.decode_layer2(l1) 
        #l3 = self.decode_layer3(l2) 
        
        return self.recon_layer(l2) 
    
    
class Deep_Autoencoder(tf.keras.Model): 
   
      
    def __init__(self, 
                 n_dims =[5570, 2785, 500], 
                 name ='Deep_Autoencoder_class_DAE', 
                 **kwargs): 
        super(Autoencoder, self).__init__(name = name, **kwargs) 
        self.n_dims = n_dims 
        self.encoder = Encoder([n_dims[0], n_dims[1], n_dims[2]])
        self.decoder = Decoder([n_dims[1], n_dims[0], n_dims[0]])
          
    @tf.function         
    def call(self, inputs): 
        x = self.encoder(inputs) 
        return self.decoder(x) 
    
    






training_epochs = 100
batch_size = 16
display_step = 1



optimizer = tf.optimizers.Adam(learning_rate = 0.00001) 




encoder = Encoder([5570, 2785, 256])
dim_reduction_model = Deep_Autoencoder([5570, 2785, 256]) 
dim_reduction_model.compile(optimizer = optimizer,  loss ='mse') 
history = dim_reduction_model.fit(x_train,x_train, batch_size = batch_size, epochs = training_epochs, validation_data=(x_test, x_test), shuffle = True)


#for saving purposes
# print('Saving the model...')
# dim_reduction_model.save('/content/drive/My Drive/Gene_Chip_Data/bioinformatics_project_data/dim_reduction_model/')
# print('model weights saved!')




#plotting the results

plt.ion()
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(111)
plt.plot(history.history['loss'],label= ' Training MSE')
plt.plot( history.history['val_loss'] ,color='red',label='Validation MSE')
ax1.set_title("Training and Testing Loss")
#    line1, = ax1.plot(test[1])
#    line2, = ax1.plot(ts_p * 0.6)
   
plt.legend(loc='upper right')
#    plt.ylim(-1, 2.0)
plt.show()



#using the encoder part to reduce the whole dataset and save it for a later classification

X = scale(X)
X = dim_reduction_model.encoder(X)

cols = []
cols = [ 'feature_'+str(i+1) for i in range(X.shape[1])]
cols.append('labels')
reduced_ds = np.c_[X,Y]

reduced_ds = pd.DataFrame(reduced_ds)
reduced_ds.columns = cols

reduced_ds.to_csv('reduced_ds_500_sm.csv', index = False)

print('done!')