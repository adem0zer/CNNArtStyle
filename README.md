# Numpy CNN
A numpy based CNN implementation for classifying images.

## Usage

Follow the steps listed below for using this repository after cloning it.  
For examples, you can look at the code in [fully_connected_network.py](https://github.com/160201024/CNNArtStyle/blob/master/layers/fully_connected.py) and [cnn.py](https://github.com/160201024/CNNArtStyle/blob/master/cnn.py).  
Data : https://www.kaggle.com/c/painter-by-numbers/data

The directory structure looks as follows 
- root
    * data\
        
    * layers\
    * loss\
    * utilities\
    * cnn.py
    * fully_connected_network.py
    
---  

1) Import the required layer classes from layers folder, for example
    ```python
    from layers.fully_connected import FullyConnected
    from layers.convolution import Convolution
    from layers.flatten import Flatten
    ```
2) Import the activations and losses in a similar way, for example
    ```python
    from layers.activation import Elu, Softmax
    from loss.losses import CategoricalCrossEntropy
    ```
3) Import the model class from utilities folder
    ```python
    from utilities.model import Model
    ```
4) Create a model using Model and layer classes
    ```python
    model = Model(
        Convolution(filters=5, padding='same'),
        Relu(),
        Convolution(filters=5, padding='same'),
        Relu(),
        Convolution(filters=5, padding='same'),
        Relu(),
        Convolution(filters=5, padding='same'),
        Relu(),
        Convolution(filters=5, padding='same'),
        Relu(),
        Pooling(mode='max', kernel_shape=(2, 2), stride=2),
        Flatten(),
        FullyConnected(units=4),
        FullyConnected(units=4),
        FullyConnected(units=4),
        Softmax(),
        name='cnn5'
    )
    ```
5) Set model loss
    ```python
    model.set_loss(CategoricalCrossEntropy)
    ```
6) Train the model using
    ```python
    model.train(data, labels)
    ```
    * set load_and_continue = True for loading trained weights and continue training
    * By default the model uses AdamOptimization with AMSgrad
    * It also saves the weights after each epoch to a models folder within the project
7) For prediction, use
    ```python
    prediction = model.predict(data)
    ```
8) For calculating accuracy, the model class provides its own function
    ```python
    accuracy = model.evaluate(data, labels)
    ```
9) To load model in a different place with the trained weights, follow till step 5 and then
    ```python
    model.load_weights()
    ```
    Note: You will have to have similar directory structure.


---


The CNN implemented here is based on [Andrej Karpathy's notes](http://cs231n.github.io/convolutional-networks/)
