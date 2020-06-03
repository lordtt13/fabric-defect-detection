# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 22:15:13 2019

@author: tanma
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from utils import train_val_generator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Input,Dropout


train_path = './train'
test_path ='./test'

train_gen, val_gen = train_val_generator(16,train_path,test_path)
input_shape = (64,64,3)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  pass

X_in = Input(input_shape)
X = Conv2D(64, 3, activation = 'relu',padding = 'same')(X_in)
X = MaxPool2D(pool_size = (2,2))(X)
X = Dropout(0.5)(X)
X = Conv2D(32, 3, activation = 'relu',padding = 'same')(X)
X = MaxPool2D(pool_size = (2,2))(X)
X = Dropout(0.5)(X)
X = Conv2D(16, 3, activation = 'relu',padding = 'same')(X)
X = MaxPool2D(pool_size = (2,2))(X)
X = Flatten()(X)
X = Dense(32,activation = 'relu')(X)
X = Dropout(0.5)(X)
X_out = Dense(2, activation = 'softmax')(X)
tr_model = Model(X_in, X_out)

checkpoint = ModelCheckpoint('vanilla.h5', monitor = 'val_accuracy', verbose = 1,save_best_only = True)
early_stop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 5,verbose = 1,mode = 'auto')

tr_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# train the model
history = tr_model.fit(
                train_gen,
                steps_per_epoch = 100,
                epochs = 25,
                validation_data = val_gen,
                validation_steps = 250,
                callbacks = [checkpoint,early_stop])

# plot the results
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.savefig('vanilla.jpg')

tr_model.save('vanilla_final.h5')