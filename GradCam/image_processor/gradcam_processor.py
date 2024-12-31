
# gradcam_processor.py
import os
import io
import base64
import numpy as np
from PIL import Image
from matplotlib import cm
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from django.conf import settings
from .model_loader import model, global_model, CLASS_INDEX

gradcam = Gradcam(model, clone=False)

def process_image_gradcam(image_path, intensity):
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
        print(f"Error processing the image (GradCAM): {e}")
        return None, "Error processing the image"
