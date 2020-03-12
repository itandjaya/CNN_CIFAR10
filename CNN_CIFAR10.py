## CNN_CIFAR10.py


from __future__ import division, absolute_import, print_function, unicode_literals;

import os;
import tensorflow as tf;
import numpy as np;
import matplotlib.pyplot as plt;    #   To display image/plot.

from CNN_Model import CNN_Model;
from tensorflow.keras import datasets, layers, models;
from tensorflow.keras.layers import Dense, Flatten, Conv2D;
from tensorflow.keras.regularizers import l1 as l1_reg, l2 as l2_reg;
from tensorflow import keras;
from random import randint;

## Import datasets locally
#from import_export_data import import_data_from_files;


## Classification labels:
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'];

def display_random_images(x_train, y_train):
    
    for i in range(25):
        idx =   randint(0, 10000);

        plt.subplot(5, 5, 1+i);
        
        ## Removes all ticks and grids on plot.
        plt.xticks([]);
        plt.yticks([]);
        plt.grid(False);

        plt.imshow(x_train[idx], cmap=plt.cm.binary);
        plt.xlabel(class_names[y_train[idx][0]]);

    plt.show();

    return;

def main():

    # Importing data from keras dataset, instead from local persistence.
    # train_ds and test_ds are in combined in single batch.
    train_ds, test_ds       =   datasets.cifar10.load_data();
    train_ds, test_ds       =   [train_ds], [test_ds];
    
    ## For partitioning data into 5 batches, use below:
    #train_ds, test_ds       =   import_data_from_files();

    ## display 5x5 random images to check if data is imported correctly.
    # train_images, train_labels  =   train_ds[0][0], train_ds[0][1];
    # display_random_images(train_images, train_labels);

    ## Initialize ConvNets model.
    ## a_batch_image_data: a batch of image data used to set 
    ##      feature normalization parameters.
    model   =   CNN_Model(  a_batch_image_data = train_ds[0][0]);

    x_test, y_test  =   test_ds.pop();

    ## Train in batches.

    for (x_train, y_train) in train_ds:

        model.train_NN(x_train, y_train, x_test, y_test);   #train data/batch.
    
    ## Test accuracy with test dataset.
    model.test_accuracy(  x_test, y_test);

    return 0;


if __name__ == '__main__':      main();
