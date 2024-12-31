import tensorflow as tf
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import io, base64
from django.conf import settings
from matplotlib import cm
from .model_loader import model, global_model, CLASS_INDEX  # Ensure model_loader is correctly set up

# Initialize Saliency
saliency = Saliency(global_model)

def process_image_integrated_gradients(image_path, intensity):
    try:
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        img = Image.open(full_image_path).resize((224, 224)).convert('RGB')
        img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        # Get the top predicted class
        top_pred = np.argmax(global_model.predict(img_array))
        score = CategoricalScore([top_pred])

        # Generate the saliency map
        saliency_map = saliency(score, img_array)

        # Normalize and overlay the saliency map on the original image
        saliency_map = normalize(saliency_map[0])
        heatmap = np.uint8(cm.jet(saliency_map[..., 0])[..., :3] * 255)
        heatmap_img = Image.fromarray(heatmap).resize((224, 224)).convert("RGBA")

        img_original = Image.open(full_image_path).resize((224, 224)).convert("RGBA")
        blended_img = Image.blend(img_original, heatmap_img, alpha=intensity / 100.0)
        blended_img = blended_img.convert("RGB")  # Convert from RGBA to RGB
        buffer = io.BytesIO()
        blended_img.save(buffer, format="JPEG")

        image_data = buffer.getvalue()
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
        return image_data_url, CLASS_INDEX[str(top_pred)][1]

    except Exception as e:
        print(f"Error processing the image (Integrated Gradients): {e}")
        return None, "Error processing the image"
