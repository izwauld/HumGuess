from util_functs import *
from pathlib import Path
from IPython.display import Audio
import time
import os
import shutil
from os.path import isdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import h5py

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Activation
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dense
from keras import backend as K

%reload_ext autoreload
%autoreload 2
%matplotlib inline
%pylab inline

np.random.seed(5)

num_classes=5
hf = h5py.File("data.h5", "r")
X_train = hf.get("train_images")
y_train = hf.get("train_labels")
X_train = np.array(X_train)
y_train = np.array(y_train)
y_train_oh = one_hot_encode(y_train.tolist(), num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), 
                 activation='relu', input_shape=X_train[1].shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.20))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.50))
model.add(Dense(5, activation='softmax'))


model.summary()
model.compile(loss = "categorical_crossentropy", optimizer = 'adam',
              metrics=['accuracy'])
model.fit(X_train, y_train_oh, batch_size=16, epochs=150, verbose=1, validation_split=0.20)

#96% accuracy in test mode
smodel.evaluate(X_train, y_train_oh, batch_size=16, verbose=1)
model.save('')
