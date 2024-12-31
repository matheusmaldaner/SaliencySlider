import requests
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from vis.utils import utils

# Load the model once
global_model = VGG19(weights='imagenet')
global_model.layers[-1].activation = tf.keras.activations.linear
model = utils.apply_modifications(global_model)

# Load class index
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
response = requests.get(url)
CLASS_INDEX = response.json()