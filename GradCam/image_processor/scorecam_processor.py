# score_cam_processor.py

import os
import io
import base64
import numpy as np
from PIL import Image
from matplotlib import cm
import traceback

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input

from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore

from django.conf import settings
from .model_loader import global_model, CLASS_INDEX


# Initialize ScoreCAM
scorecam = Scorecam(global_model, clone=False)

def process_image_scorecam(image_path, intensity):
    """
    Score-CAM uses forward-pass scores to generate localization maps, 
    removing the need for explicit gradient signals.
    """
    try:
        # Normalize path - replace backslashes with forward slashes
        image_path = image_path.replace('\\', '/').strip()
        
        # Construct full path using os.path.join which handles OS-specific path separators
        full_image_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, image_path))
        
        print(f"[Score-CAM] Loading image from path: {full_image_path}")

        # 1. Load & preprocess
        img = Image.open(full_image_path)
        original_size = img.size
        img_copy = img.resize((224, 224)).convert('RGB')

        x = np.expand_dims(img_to_array(img_copy), axis=0)
        x = preprocess_input(x)

        # 2. Predict top class
        predictions = global_model.predict(x)
        top_pred = np.argmax(predictions[0])
        predicted_class_name = CLASS_INDEX[str(top_pred)][1]

        # 3. Score function for top class
        score = CategoricalScore([top_pred])

        # 4. Generate Score-CAM
        cam = scorecam(score, x, penultimate_layer=-1)
        cam = normalize(cam)

        # 5. Convert to heatmap
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        heatmap = Image.fromarray(heatmap).resize(original_size, Image.LANCZOS).convert("RGBA")

        # 6. Blend
        alpha = max(0, min(100, intensity)) / 100.0
        blended_img = Image.blend(img.convert("RGBA"), heatmap, alpha=alpha).convert("RGB")

        # 7. Encode to base64
        buffer = io.BytesIO()
        blended_img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image_data_url = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

        print(f"[Score-CAM] Predicted: {predicted_class_name}")
        return image_data_url, predicted_class_name

    except Exception as e:
        print(f"Error processing the image (Score-CAM): {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None, f"Error processing the image: {str(e)}"
