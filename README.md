# CNN_CIFAR10
Classify images from the CIFAR-10 image dataset. Includes image data pre-processing and training the ConvNets model.

Note:
1. Input features will be normalized before fed into the model.
2. Input dataset will be in one-hot data format.
3. Data will be randomly shuffled, then split into 5 batches. 10k data/batch.
4. Validation data size: 10k.

Classification will be as follows:

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'];


Below is summary of the ConvNets stacks:

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0
_________________________________________________________________
batch_normalization (BatchNo (None, 16, 16, 32)        128
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0
_________________________________________________________________
batch_normalization_1 (Batch (None, 8, 8, 64)          256
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 128)         73856
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0
_________________________________________________________________
batch_normalization_2 (Batch (None, 4, 4, 128)         512
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0
_________________________________________________________________
batch_normalization_3 (Batch (None, 2048)              8192
_________________________________________________________________
dense (Dense)                (None, 64)                131136
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 234,122
Trainable params: 234,122
Non-trainable params: 0
