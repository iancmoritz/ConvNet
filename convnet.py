#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
tf.__version__


# In[2]:


#PARAMETERS


# In[3]:


#PREPROCESSING
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(validation_split=0.2)
##
training_generator = datagen.flow_from_directory('data/training', target_size=(75, 75), batch_size=1000, shuffle=True, seed=42, subset='training')
validation_generator = datagen.flow_from_directory('data/training', target_size=(75, 75), batch_size=1000, shuffle=True, seed=42, subset='validation')
testing_generator = datagen.flow_from_directory('data/testing', target_size=(75, 75), batch_size=1000, shuffle=True, seed=42)


# In[4]:


#CREATE MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense

model = Sequential()

#Layer 1
model.add(Conv2D(64, (3,3), strides=(1,1), activation="relu"))

#Layer 2 
model.add(MaxPooling2D())

#Layer 3
model.add(Conv2D(128, (3,3), strides=(1,1), activation="relu"))

#Layer 4
model.add(BatchNormalization())

#Layer 5
model.add(Dropout(0.3))

#Layer 6
model.add(MaxPooling2D())

#Layer 7
model.add(Flatten())

#Layer 8
model.add(Dense(256, activation="relu"))

#Layer 9
model.add(Dense(131, activation="softmax"))


# In[7]:


#COMPILE MODEL 
model.compile(loss="categorical_crossentropy",
              optimizer="adam")


# In[ ]:


model.fit(
        training_generator,
        steps_per_epoch=training_generator.samples // 1000, #batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 1000,
        verbose=1)
model.save_weights('first_try.h5')  # always save your weights after training or during training


# In[ ]:




