## import_data.py

#import numpy as np;
import os;
import numpy as np;
import pickle;
import CNN_Model;


CURR_PATH           =   os.getcwd();
DATA_PATH           =   CURR_PATH + r'/data/images/cifar-10-batches-py/';
ROWS, COLS, DIMS    =   32, 32, 3;

NORMALIZED_DATA_FILE    =   DATA_PATH + r'normalized_data.p';

def unpickle(file):
    ## Input:
    ##      file:   str: filename to be imported & unpacked.
    ##      return: dict: dictionary of batch dataset.

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes');
    return dict;

def import_data_from_files(    file_path = DATA_PATH):
    ## Input:
    ##      file_path:   str: directory path of data files.
    ##      return: tuple: (train_dataset, test_dataset).

    file_names  =   [];
    train_data  =   [];
    test_data   =   [];

    for (dirpath, dirnames, filenames) in os.walk(file_path):
        
        for filename in filenames:


            if  not filename.lower().startswith("data_batch") and \
                not filename.lower().startswith("test_batch"):
                continue;

            ## Import data from files.   
            dict_data   =   unpickle(   dirpath + filename);
            num_samples =   len(dict_data[b'data']);

            ## Data in dictionary format.
            batch_ID    =   dict_data[b'batch_label'];      # Batch ID: 1 - 5.
            fn          =   dict_data[b'filenames'];        # image filenames.
            labels      =   np.array(dict_data[b'labels']); # Classification labels.

            ## Image data in numpy dimensions: 
            ##  num_samples, RGB, rows, colors: 10k, 3, 32, 32.     
            images      =   dict_data[b'data'].reshape(num_samples, DIMS, ROWS, COLS);


            #Dimension order: ith_sample, rows, cols, colors.
            images  =   images.transpose(0, 2, 3, 1);  


            if  filename.startswith("data_batch"):
                train_data.append(  (images, labels));

            elif filename.startswith("test_batch"):
                test_data.append(  (images, labels));

    return  train_data, test_data;