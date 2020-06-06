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

# relative paths to the folders containing images
train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

# prepocess images and create batches of the data
train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224, 224), classes=['engine','ship'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['engine','ship'], batch_size=8)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224, 224), classes=['engine','ship'], batch_size=20, shuffle=False)

print(train_batches.class_indices)

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
# train the model
model.fit_generator(generator=train_batches, steps_per_epoch=8, validation_data=valid_batches, validation_steps=1, epochs=30, verbose=2)

### testing the model
# Get the labels from the test batch
test_labels = test_batches.classes
predictions = model.predict_generator(test_batches, steps=1, verbose=0)

# plotting predictions
def plot_confusion_matrix(cm, classes,
            normalize=False,
            title='Confusion matrix',
            cmap=plt.cm.get_cmap("Blues")):

# This function prints and plots the confusion matrix.
# Normalization can be applied by setting `normalize=True`.

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
cm_plot_labels = ['engine', 'ship']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')