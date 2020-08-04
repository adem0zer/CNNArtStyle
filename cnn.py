from layers.fully_connected import FullyConnected
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.flatten import Flatten
from layers.activation import Elu, Softmax, Relu

from utilities.model import Model
from utilities.resizeImg import resize

from loss.losses import CategoricalCrossEntropy
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd

import numpy as np
np.random.seed(0)

#Symbolism
#Impressionism
#Realism
#Expressionism


if __name__ == '__main__':
    path="D:/Downloads/Compressed/train/"
    img_resize = 32
   
    style_codes = { "Symbolism":       [1.0,0.0,0.0,0.0],
                    "Impressionism":   [0.0,1.0,0.0,0.0],
                    "Realism":         [0.0,0.0,1.0,0.0],
                    "Expressionism":   [0.0,0.0,0.0,1.0]}


    df = pd.read_csv("train_latest.csv",sep=',')
    df_filename = df[['filename']]
    df_style = df[['style']]

    train_data = np.full((df_filename.size,img_resize,img_resize,3), 0)
    train_label = np.full((df_filename.size,4),0)
    imgFile = "train/29331.jpg"
    predictData = np.full((1,img_resize,img_resize,3), 0)
    predictData[0] = resize(imgFile,img_resize,img_resize)
    predictData[0] = predictData[0]/255

    print(predictData)
    for x in range(0,len(df_filename.index)):
        img_file=path+df_filename['filename'][x]
        train_data[x]=resize(img_file,img_resize,img_resize)
        temp_label=style_codes[df_style['style'][x]]
        train_label[x]=temp_label
        print(img_file," - ",temp_label)
    
    train_data=train_data/255

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

    #model.set_loss(CategoricalCrossEntropy)

    #model.train(train_data, train_label.T, epochs=2) # set load_and_continue to True if you want to start from already trained weights
    model.load_weights() # uncomment if loading previously trained weights and comment above line to skip training and only load trained weights.

    print('Testing accuracy = {}'.format(model.evaluate(train_data, train_label)))
    print('Testing accuracy = {}'.format(model.predict(predictData)))
