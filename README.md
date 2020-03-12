# CNN_CIFAR10
Classify images from the CIFAR-10 image dataset. Includes image data pre-processing and training the ConvNets model.

File information:
acc_plot.png  - Plot of train vs. test_validation accuracy. test_acc converges ~ 0.78%
CNN_CIFAR10.py
CNN_Model.py
CNN_Model_Stacks.txt  - The summary of the ConvNets network.
import_export_data.py - Script to import and process the data.
Note:
1. Input features will be normalized before fed into the model.
2. Input dataset will be in one-hot data format.
3. Data will be randomly shuffled, then split into 5 batches. 10k data/batch.
4. Validation data size: 10k.

Classification will be as follows:

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'];


