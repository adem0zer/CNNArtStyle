from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Elu, Softmax

from utilities.model import Model
from utilities.resizeImg import resize

from loss.losses import CategoricalCrossEntropy
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd

import numpy as np
np.random.seed(0)

model = Model(
    Convolution(filters=5, padding='same'),
    Elu(),
    Pooling(mode='max', kernel_shape=(2, 2), stride=2),
    Flatten(),
    FullyConnected(units=4),
    Softmax(),
    name='cnn5'
)


model.load_weights() # uncomment if loading previously trained weights and comment above line to skip training and only load trained weights.
        
print('Testing accuracy = ')
print(model.predict(cv2.imread('train/4.jpg'))



