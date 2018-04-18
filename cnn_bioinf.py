## Convolutional Neural Network - Ana1
"""
Created on Sat Mar 31 20:00:30 2018

@author: anaro
"""

# Installing Theano

# Installing Tensorflow

# Installing Keras

# Installing PIL

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras import backend as K
import keras.optimizers
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
import csv

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution #Kernel: 3x3 #Funcion de activacion: ReLU
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection 
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 4, activation = 'softmax'))

# Compiling the CNN
Optimizer = keras.optimizers.Adam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  decay=0.
classifier.compile(loss='categorical_crossentropy',
              optimizer=Optimizer,
              metrics=['categorical_accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory('pathwork/train_set',
                                                 target_size = (64,64),
                                                 batch_size = 32
                                                 )

test_set = test_datagen.flow_from_directory('pathwork/val_set',
                                            target_size = (64,64),
                                            batch_size = 32
                                            )

classifier.fit_generator(train_set,
                         samples_per_epoch = 800,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 200)
                         #use_multiprocessing=True

## Part 3 - Predict labels for new test sets 
file=open(pathwork + 'predict_set\\Labels.csv','r')
fileLabels=csv.reader(file)
labels=list()
for line in fileLabels:
    labels=line
file.close()
pred_labels=[int(i) for i in labels]
    
filesImages=os.listdir()
del filesImages[filesImages.index('Labels.csv')]
labelsCNN=list(range(len(filesImages)))
for i in range(len(filesImages)):
    test_image = image.load_img(filesImages[i], target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    indices=train_set.class_indices
    labelsCNN[i]=result.argmax()
       
accuracy=accuracy_score(pred_labels, labelsCNN)
recall=recall_score(pred_labels,labelsCNN,average='weighted')   
precision=precision_score(pred_labels,labelsCNN,average='weighted')   
    
#if result[0][0] == 0:
#prediction = 'flower'
#if result[0][0] == 1:
#prediction = 'fruit'
#if result[0][0] == 2:
#prediction = 'leaf'
#if result[0][0] == 3:
#prediction = 'trichoma'

