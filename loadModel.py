#handy tools
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

# Tf related imports
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow import saved_model
from keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('savedModel/1/')

test_path = 'data/test'
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224, 224), classes=['engine', 'ship'], batch_size=20, shuffle=False)

#predict using custom model and store predictions
predictions = model.predict_generator(test_batches, steps=1, verbose=2)

#predict and plot outcome
test_labels = test_batches.classes

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
