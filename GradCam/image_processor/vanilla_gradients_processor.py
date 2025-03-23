# vanilla_gradients_processor.py

import os
import io
import base64
import numpy as np
from PIL import Image
from matplotlib import cm
import traceback

# TensorFlow / Keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

# tf-keras-vis for Vanilla Gradients (Saliency)
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore

# Django settings
from django.conf import settings

# Your models and class index (adjust the import as needed)
from .model_loader import global_model, CLASS_INDEX

# Instantiate Saliency for Vanilla Gradients
saliency = Saliency(global_model)

def process_image_vanilla_gradients(image_path, intensity):
    """
    Applies Vanilla Gradients (Saliency) on the top-predicted class and blends
    the resulting heatmap with the original image.
    
    :param image_path: The relative path to the image in MEDIA_ROOT.
    :param intensity: An integer 0–100 controlling heatmap opacity.
    :return: (base64_image, predicted_class_name) or (None, error_string)
    """
    try:
        # Normalize path - replace backslashes with forward slashes
        image_path = image_path.replace('\\', '/').strip()
        
        # Construct full path using os.path.join which handles OS-specific path separators
        full_image_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, image_path))
        
        print(f"[Vanilla Gradients] Loading image from path: {full_image_path}")

        # 1. Load and preprocess the image
        img = Image.open(full_image_path)
        original_size = img.size

        # Resize to 224x224 for VGG19 and convert to RGB
        img_copy = img.resize((224, 224)).convert('RGB')
        img_array = np.expand_dims(img_to_array(img_copy), axis=0)
        img_array = preprocess_input(img_array)

        # 2. Predict top class
        predictions = global_model.predict(img_array)
        top_pred_index = np.argmax(predictions[0])

        # 3. Define the score function for that top class
        score = CategoricalScore([top_pred_index])

        # 4. Generate Vanilla Gradients (Saliency) maps
        # result shape: [1, H, W], because it's a single-channel saliency
        saliency_map = saliency(score, img_array)
        # saliency_map[0] has shape (224, 224)

        # 5. Normalize the saliency map
        saliency_map = normalize(saliency_map)

        # 6. Create a color heatmap from the single-channel map
        # We take saliency_map[0], which is shape (224, 224)
        heatmap = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)
        heatmap_resized = Image.fromarray(heatmap).resize(original_size, Image.LANCZOS).convert("RGBA")

        # 7. Blend heatmap with the original image
        img_original = img.convert("RGBA")
        alpha = max(0, min(100, intensity)) / 100.0
        blended_img = Image.blend(img_original, heatmap_resized, alpha=alpha).convert("RGB")

        # 8. Convert blended image to base64
        buffer = io.BytesIO()
        blended_img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')

        # 9. Return the data URL along with the predicted class label
        predicted_class_name = CLASS_INDEX[str(top_pred_index)][1]
        print(f"Predicted class: {predicted_class_name}")
        print(f"Generated image_data_url (truncated): {image_data_url[:100]}...")
        return image_data_url, predicted_class_name

    except Exception as e:
        print(f"Error processing the image (Vanilla Gradients): {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None, f"Error processing the image: {str(e)}"
