import json
from vis.utils import utils
from matplotlib import cm
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import io, base64
import requests
from PIL import Image, ImageFilter
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from django.conf import settings
import os


global_model = VGG19(weights='imagenet')
global_model.layers[-1].activation = tf.keras.activations.linear
model = utils.apply_modifications(global_model)


url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
response = requests.get(url)
CLASS_INDEX = response.json()


gradcam = Gradcam(model, clone=False)

def process_image(image_path, intensity):
    # Now image_path should be something like 'user_images/20Ounce_NYAS-Apples2.png'
    full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
    print(f"Loading image from path: {full_image_path}")

    try:
        img = Image.open(full_image_path)
        original_size = img.size  
        img_copy = img.resize((224, 224)).convert('RGB')

        img_array = img_to_array(img_copy)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = global_model.predict(img_array)
        top_pred = np.argmax(predictions[0])

        classlabel = [CLASS_INDEX[str(i)][1] for i in range(len(CLASS_INDEX))]

        score = CategoricalScore([top_pred])
        cam = gradcam(score, img_array, penultimate_layer=-1)
        cam = normalize(cam)

        heatmap_resized = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        heatmap_resized = Image.fromarray(heatmap_resized).resize(original_size, Image.LANCZOS)

        img_original = img.convert("RGBA")
        heatmap_resized = heatmap_resized.convert("RGBA")

        intensity = max(0, min(100, intensity))
        alpha = intensity / 100.0

        blended_img = Image.blend(img_original, heatmap_resized, alpha=alpha).convert("RGB")

        buffer = io.BytesIO()
        blended_img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
        print(f"Generated image_data_url (truncated): {image_data_url[:100]}...")
        return image_data_url, classlabel[top_pred]

    except Exception as e:
        print(f"Error processing the image: {e}")
        return None, "Error processing the image"

# def process_image(image_url, intensity):
#     # Load model
#     model = VGG19(weights='imagenet')
#     model.summary()  # For debugging, to see model layers

#     # Fetch image from URL and preprocess
#     response = requests.get(image_url)
#     img = Image.open(io.BytesIO(response.content))
#     img = img.resize((224, 224))  # Resize for the model input
#     img_array = img_to_array(img)  # Convert to array
#     img_array = np.expand_dims(img_array, axis=0)  # Make 'batch' of 1
#     img_array = preprocess_input(img_array)  # Preprocess the input

#     # Predictions
#     predictions = model.predict(img_array)
#     top_pred = np.argmax(predictions[0])

#     # GradCAM
#     gradcam = Gradcam(model, model_modifier=None, clone=False)
#     # Generate heatmap with GradCAM
#     cam = gradcam(top_pred, img_array, penultimate_layer=-1)  # Use appropriate layer
#     heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Get heatmap

#     # Merge heatmap with original image
#     heatmap = Image.fromarray(heatmap)
#     merged_img = Image.blend(img.convert("RGBA"), heatmap.convert("RGBA"), alpha=0.5)
    
#     # Convert to data URL
#     buffer = io.BytesIO()
#     merged_img.save(buffer, format="JPEG")
#     image_data = buffer.getvalue()
#     image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')

#     return image_data_url