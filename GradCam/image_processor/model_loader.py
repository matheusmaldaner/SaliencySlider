import requests
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# 1) Load VGG19 model (with imagenet weights)
global_model = VGG19(weights='imagenet')

# 2) Remove or replace the final softmax activation (important for many saliency methods)
modifier = ReplaceToLinear()
modifier(global_model)  # In-place modification

# If you want a separate reference, you can do:
model = global_model

# 3) Load class index (e.g. from Raghakot's GitHub)
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
response = requests.get(url)
CLASS_INDEX = response.json()
