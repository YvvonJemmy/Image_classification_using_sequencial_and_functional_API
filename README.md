# Convolution neural networks in tensorflow
The code classifies images of cats and dogs
# Requirements
- Have tensorflow installed
- Have python installed 
# Libraries used
- numpy library for arrays
- random library for picking a random image
- sequencial for making a sequencial model
- model for making a functional model
- Input,Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, BatchNormalization, Reshape for making convolution layers
# Number of images
- There are 2000 images for the training set and 400 images on the testing set
# models used
Two models have been used
- sequencial model
- functional model
# working on the dataset
The images have been reshaped into a shape of 100*100*3
The images have then be rescaled by dividing them with 255
# Model building
### sequential model
- Adds a 2D convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation function. The input_shape parameter specifies the shape of the input image, which is 100x100 pixels with 3 color channels (RGB).
- Adds a max pooling layer with a pool size of 2x2 pixels.
- Adds another 2D convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation function.
- Adds another max pooling layer with a pool size of 2x2 pixels.
- Flattens the output from the convolutional layers into a 1D array, so it can be passed to a fully connected layer.
- Adds a fully connected (Dense) layer with 64 units and a ReLU activation function.
- Adds another fully connected layer with a single unit and a sigmoid activation function. This layer is used for binary classification tasks.

Overall, this CNN has two convolutional layers, two max pooling layers, and two fully connected layers. It is designed to take in color images of size 100x100 pixels and output a binary classification prediction.

The sequential model  is compiled using the binary cross-entropy loss function and the Adam optimizer. The model's performance will be evaluated based on the accuracy metric.

The binary cross-entropy loss function is commonly used in binary classification problems where the output is either 0 or 1. It measures the difference between the predicted probability distribution and the actual probability distribution. The Adam optimizer is a popular optimizer used to update the model weights based on the calculated gradients.

The fit method is applied. The x_train and y_train arguments represent the input data and labels for the training set, respectively.

The epochs argument specifies the number of times the entire training dataset will be processed by the model during training. The batch_size argument defines the number of samples that will be propagated through the model at once. The code has an epochs value of 10.

During training, the model will update its weights based on the calculated gradients using the optimizer and loss function defined during compilation.

### Accuracy of the model
The model has an accuracy of 0.7275 on the training set and an accuracy of 0.6675 on the testing set
### Functional model
The input layer is defined using the Input class, specifying a shape of (100, 100, 3) which means it expects input images of size 100x100 pixels with 3 color channels.

The first convolutional layer is defined using Conv2D class with 32 filters of size (3,3), using the 'relu' activation function. The input to this layer is the input_layer.

The first pooling layer is defined using MaxPooling2D class with a pooling size of (2,2). The input to this layer is the conv_layer1.

The second convolutional layer is defined using Conv2D class with 32 filters of size (3,3), using the 'relu' activation function. The input to this layer is the pooling_layer1.

The second pooling layer is defined using MaxPooling2D class with a pooling size of (2,2). The input to this layer is the conv_layer2.

The flatten layer is defined using the Flatten class. This layer flattens the output of the previous layer into a 1D tensor.

The first dense layer is defined using Dense class with 64 units and 'relu' activation function. The input to this layer is the flatten_layer.

The output layer is defined using Dense class with 1 unit and 'sigmoid' activation function. The input to this layer is the dense_layer1.

Finally, the Model class is used to create a model instance with the input layer and output layer as arguments.

The sequential model  is compiled using the binary cross-entropy loss function and the Adam optimizer. The model's performance will be evaluated based on the accuracy metric.

The binary cross-entropy loss function is commonly used in binary classification problems where the output is either 0 or 1. It measures the difference between the predicted probability distribution and the actual probability distribution. The Adam optimizer is a popular optimizer used to update the model weights based on the calculated gradients.

The fit method is applied. The x_train and y_train arguments represent the input data and labels for the training set, respectively.

The epochs argument specifies the number of times the entire training dataset will be processed by the model during training. The batch_size argument defines the number of samples that will be propagated through the model at once. The code has an epochs value of 10.

During training, the model will update its weights based on the calculated gradients using the optimizer and loss function defined during compilation.

### Accuracy of the model
The model has an accuracy of 0.9855 on the training set and an accuracy of 0.6325 on the testing set

# OVERVIEW
The models are overfitting the data. This is mostly due to using less data to train them.





