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

mobile = keras.applications.mobilenet.MobileNet()

# prepare an image for prediction
# @param file name as string
def prepare_image(file):
    img_path = './images/'
    #resize image to fit the input of the model.
    img = image.load_img(img_path + file, target_size=(224, 224))
    #convert image to list/array
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

#preprocess a image and store it.
prepped_image = prepare_image("1.jpg")
#predict using mobileNet and store predictions
predictions = mobile.predict(prepped_image)
#decode predictions
results = imagenet_utils.decode_predictions(predictions)

print(results)