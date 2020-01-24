#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import os, os.path
import keras
import matplotlib.pyplot as pyplot
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.applications import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras import Sequential
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
import PIL.Image as Image
from keras.models import model_from_json, load_model
from cv2 import cv2
import glob
#import RPi.GPIO as GPIO
import time


# In[51]:


image_size = 256


# In[52]:


def model_CNN():
    #vgg = VGG16(include_top=False, pooling='avg', weights='imagenet', input_shape=(224,224,3))
    cnn = Sequential()# Convolution - extracting appropriate features from the input image.
    cnn.add(Conv2D(filters=32, 
                    kernel_size=(2,2), 
                    strides=(1,1),
                    padding='same',
                    input_shape=(image_size,image_size,3),
                    data_format='channels_last'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2),
                        strides=2))
    cnn.add(Conv2D(filters=64,
                    kernel_size=(2,2),
                    strides=(1,1),
                    padding='valid'))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2),
                        strides=2))
    cnn.add(Flatten())        
    cnn.add(Dense(64))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Dense(10))
    cnn.add(Activation('softmax'))
    return cnn


# In[53]:


def get_data():
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_datagen  = ImageDataGenerator(preprocessing_function=preprocess_input)

    batch = 32

    train_generator = train_datagen.flow_from_directory(
                                                        directory="/Users/hashad/Documents/AUC/Fall 2019/Computer Vision/Train",
                                                        target_size=(image_size,image_size),
                                                        color_mode='rgb',
                                                        batch_size=batch,
                                                        class_mode="categorical",
                                                        shuffle=True,
                                                        seed=42)


    valid_generator = valid_datagen.flow_from_directory(
                                                        directory="/Users/hashad/Documents/AUC/Fall 2019/Computer Vision/Valid",
                                                        target_size=(image_size, image_size),
                                                        color_mode="rgb",
                                                        batch_size=8,
                                                        class_mode="categorical",
                                                        shuffle=True,
                                                        seed=42
    )

    test_generator = test_datagen.flow_from_directory(
                                                        directory="/Users/hashad/Documents/AUC/Fall 2019/Computer Vision/Test",
                                                        target_size=(image_size, image_size),
                                                        color_mode="rgb",
                                                        batch_size=1 ,
                                                        class_mode="categorical",
                                                        shuffle=False
                                                        
    )
    return train_generator,valid_generator,test_generator, batch


# In[54]:


def set_model():
    model = model_CNN()
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=[ 'accuracy'])
    return model


# In[55]:


def fit_model(train_generator, valid_generator, model):
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=3
    )
    return model, STEP_SIZE_VALID


# In[56]:


def evaluate_and_save_model(model, valid_generator,STEP_SIZE_VALID,test_generator):

    model.evaluate_generator(generator=valid_generator,
    steps=STEP_SIZE_VALID)

    
    model.save('/Users/hashad/Documents/AUC/Fall 2019/Computer Vision/final_model_3.h5')
    #return pred


# In[57]:


train_generator,valid_generator,test_generator, batch = get_data()
model = set_model()
model, STEP_SIZE_VALID =  fit_model(train_generator, valid_generator, model)
evaluate_and_save_model(model, valid_generator,STEP_SIZE_VALID,test_generator)


# In[66]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from cv2 import cv2
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(256,256)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


# In[89]:


img=cv2.imread('6.jpg')
model=load_model('final_model_3.h5')
img=cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
img=np.array([img])
prediction=model.predict(img)
for i in range(len(prediction[0])):
    if prediction[0][i]>=1:
        print(i) 


# In[81]:





# In[ ]:




