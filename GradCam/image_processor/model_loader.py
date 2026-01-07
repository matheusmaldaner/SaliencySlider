import json
import os
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

# 3) Load class index from local file (avoids network dependency on startup)
# The imagenet_class_index.json file should be in the project root directory
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_class_index_path = os.path.join(_base_dir, 'imagenet_class_index.json')

try:
    with open(_class_index_path, 'r') as f:
        CLASS_INDEX = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"imagenet_class_index.json not found at {_class_index_path}. "
        "Please ensure this file exists in the project root directory."
    )
