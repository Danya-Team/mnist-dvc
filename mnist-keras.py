from mlutils import git_version, mlversion, timestamp
import git
from datetime import datetime
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.datasets import mnist
from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import platform
import sys
import subprocess
import os
from tensorflow.python.keras.utils.vis_utils import plot_model
import matplotlib.image as mpimg
import logging
import coloredlogs

coloredlogs.install(level='DEBUG')
log = logging.getLogger('mnist-keras')

MODEL_VERSION = timestamp() + git_version() + "-V1.0"

log.warning(MODEL_VERSION)
log.error(MODEL_VERSION)

mlversion()

# sys.exit()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(
    os.getcwd() + "/datasets/mnist.npz")

train_images.shape
train_labels.shape
test_images.shape
test_labels.shape

img = train_images[100]
plt.imshow(img, cmap=plt.cm.binary)


def build_mode():
    model = keras.Sequential(name=MODEL_VERSION)
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='models1.png',
               show_shapes=True, show_layer_names=True)
    # plt.imshow(mpimg.imread("models1.png"))
    return model


train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype('float32')/255
train_labels = to_categorical(train_labels)

test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32')/255
test_labels = to_categorical(test_labels)


with tf.distribute.MirroredStrategy().scope():
    model = build_mode()

model.fit(train_images, train_labels, epochs=2, batch_size=128)

model_path = os.path.join("./models/")
if not os.path.exists(model_path):
    os.makedirs(model_path)
model_file = os.path.join(model_path, MODEL_VERSION + ".h5")

model.save(model_file)
model.summary()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_loss : ", test_loss)
print("test_acc  : ", test_acc)
