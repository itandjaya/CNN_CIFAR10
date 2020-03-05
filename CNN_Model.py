## CNN_Model.py
## Convolution Nets class.

from __future__ import division, absolute_import, print_function, unicode_literals;

#import os;
import tensorflow as tf;

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D;
from tensorflow.keras import Model;

import matplotlib.pyplot as plt;    #   To display image/plot.
import numpy as np;

BATCH_SIZE  =   10000;
EPOCHS      =   50;



## Inherits tf.keras.Model
class   CNN_Model(Model):

    def __init__(self, a_batch_image_data = None):
        ## Input: set_norm_params

        super(CNN_Model, self).__init__(); 

        self.epochs         =   EPOCHS;
        self.train_history  =   [];


        ## Initializing the layers.
        #self.L1_input           =   Flatten(input_shape = (32, 32, 3));
        self.L1_conv            =   Conv2D( 32, (3,3), 
                                            activation = 'relu',
                                            input_shape = (32,32,3));        
        self.L2_pooling         =   MaxPooling2D( (2,2));

        self.L3_conv            =   Conv2D( 64, (3,3), 
                                            activation = 'relu');        
        self.L4_pooling         =   MaxPooling2D( (2,2));

        self.L5_conv            =   Conv2D( 64, (3,3), 
                                            activation = 'relu');        

        self.L6_flatten         =   Flatten();
        self.L7_dense           =   Dense(  64, activation = 'relu');
        self.L8_output_dense    =   Dense(  10);

        self.L9_output_softmax =   tf.keras.layers.Softmax();


        ## Stacking the NN layers to build a model.
        self.model = tf.keras.models.Sequential([
                                                    self.L1_conv,
                                                    self.L2_pooling,
                                                    self.L3_conv,
                                                    self.L4_pooling,
                                                    self.L5_conv,
                                                    self.L6_flatten ,
                                                    self.L7_dense,
                                                    self.L8_output_dense,
                                                    #self.L9_output_softmax,
                                                    ]);
        self.compile_model();

        self.model.summary();   #display the architecture of our model. 

        ## parameters used for normalization function.
        self.norm_mean  =   0;
        self.norm_range =   1;

        self.set_normalize_feature_params(   a_batch_image_data);
        print(self.norm_mean, self.norm_range);

        return;
    
    
    def set_normalize_feature_params(self, X_input):
        ## Sets the constant parameters used for feature normalization.
        if  X_input.any():
            self.norm_mean  =   np.mean(X_input);
            self.norm_range =   np.max(X_input) - np.min(X_input);
        return;

    def compile_model(self):
        ## Loss function: Use cross-entropy (log).
        loss_fn     =   tf.keras.losses.SparseCategoricalCrossentropy(    from_logits = True);

        ## Optimizer.
        #opt    =  tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False);
        #opt         =   tf.keras.optimizers.SGD(learning_rate=0.01);
        opt         =  tf.keras.optimizers.Adam();

        self.model.compile(     optimizer   =   opt,
                                loss        =   loss_fn,
                                metrics     =   ['accuracy']);
        
        return;

    def train_NN(self, X_input, y_output, X_test, y_test):
        # Add a channels dimension.
        #X_input =   X_input[..., tf.newaxis];

        X_input     =   self.normalize_features(    X_input);   # Normalize input features.
        X_test      =   self.normalize_features(    X_test);

        self.train_history   =   self.model.fit(     
                                            X_input,
                                            y_output, 
                                            epochs  =   self.epochs,
                                            #validation_split = 0.2,
                                            #epochs = self.epochs,
                                            #batch_size=BATCH_SIZE/5,
                                            validation_data = (X_test, y_test), 
                                            callbacks = None,                             
                                            verbose = 2);
        return self.train_history;
    
    def plot_training_model(self, history = []):

        history =   self.train_history;

        plt.plot(   history.history['accuracy'],        label='train_accuracy');
        plt.plot(   history.history['val_accuracy'],    label = 'test_accuracy');

        plt.xlabel('Epoch');
        plt.ylabel('Accuracy');
        plt.legend(loc='lower right');

        plt.ylim(   [0.3, 1]);
        plt.show();

        return;

    def predict_funct(self, X_input):
        X_input     =   X_input.astype(np.float32);
        X_input     =   self.normalize_features(    X_input);   # Normalize input values.

        # Add a channels dimension.
        #X_input =   X_input[..., tf.newaxis];  

        prob_output =   self.model.predict(X_input);
        prob_output =   self.L3_output_softmax(prob_output)[0].numpy();

        predicted_digit =   np.argmax(prob_output);

        return  predicted_digit, prob_output;

    def test_accuracy(self, X_input, y_output):
        ## Returns the % accuracy rate of the predicted output vs. data output.

        X_input     =   self.normalize_features(    X_input);   # Normalize input values.
        eval_loss, eval_accuracy  =   self.model.evaluate(     X_input,  y_output, verbose=2);
        return eval_loss, eval_accuracy ;

    def normalize_features(self, x_input):
        ## Returns normalized version of x_input.
        ## Input:
        ##      x_input: np.array: feature data to be normalized.
        ## 
        return (x_input - self.norm_mean) / self.norm_range;

    def one_hot(self, y_input):
        ##  Converts a class vector (integers) to binary class matrix.  
        ##  Input:
        ##      y_input: np.array: 1-dim vector (numpy).
        ##      
        return tf.keras.utils.to_categorical(   y_input, np.max(y_input));
