import numpy as np
import keras
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
# %matplotlib inline

# relative paths to the folders containing images
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# prepocess images and create batches of the data
train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224, 224), classes=['engine','ship'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['engine','ship'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224, 224), classes=['engine','ship'], batch_size=7, shuffle=False)

# load mobileNet model
mobile = keras.applications.mobilenet.MobileNet()

### Modifying the mobileNet model 
# copy mobileNet up untill the 6th to last layer.
x = mobile.layers[-6].output
# append a output layer
predictions = Dense(2, activation='softmax')(x)
# construct the new model
model = Model(inputs=mobile.input, outputs=predictions)
# freezing all layers accept the last 5. Only the last 5 layers will be retrained.
# experiment with this number to try and get better results.
for layer in model.layers[:-5]:
    layer.trainable = False

### retraining the model
# compile the model for training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=train_batches, steps_per_epoch=1, validation_data=valid_batches, validation_steps=1, epochs=30, verbose=2)