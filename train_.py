# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 22:15:13 2019

@author: tanma
"""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))

import matplotlib.pyplot as plt
from utils import train_val_generator
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,Input
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,EarlyStopping
from keras.optimizers import Adam


train_path = './train'
test_path ='./test'

train_gen, val_gen = train_val_generator(16,train_path,test_path)
input_shape = (128,128,3)

X_in = Input(input_shape)
X = Conv2D(64,3,activation='relu',padding='same')(X_in)
X = MaxPool2D(pool_size=(2,2))(X)
X = Conv2D(128,3,activation='relu',padding='same')(X)
X = MaxPool2D(pool_size=(2,2))(X)
X = Flatten()(X)
X = Dense(256,activation='relu')(X)
X_out = Dense(2,activation='softmax')(X)
tr_model = Model(input = X_in, output = X_out)

checkpoint = ModelCheckpoint('vanilla.h5',monitor='val_acc',verbose=1,save_best_only=True)
early_stop = EarlyStopping(monitor='val_acc',min_delta=0,patience=5,verbose=1,mode='auto')

tr_model.compile(loss='categorical_crossentropy',optimizer=Adam(1e-5),metrics=['accuracy'])

# train the model
history = tr_model.fit_generator(
                train_gen,
                steps_per_epoch=1000,
                epochs=30,
                validation_data = val_gen,
                validation_steps = 250,
                callbacks = [checkpoint,early_stop])

# plot the results
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.savefig('vanilla.jpg')

tr_model.save('vanilla_final.h5')