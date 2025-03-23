import os
import io
import base64
import numpy as np
from PIL import Image
from django.conf import settings
from matplotlib import cm
import traceback
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils import normalize
from .model_loader import model, global_model, CLASS_INDEX  # Ensure model_loader is correctly set up

# Initialize Activation Maximization
activation_maximization = ActivationMaximization(global_model)

def process_image_activation_maximization(image_path, intensity):
    try:
        # Normalize path - replace backslashes with forward slashes
        image_path = image_path.replace('\\', '/').strip()
        
        # Construct full path using os.path.join which handles OS-specific path separators
        full_image_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, image_path))
        
        print(f"[Activation Maximization] Loading image from path: {full_image_path}")
        
        img = Image.open(full_image_path)
        original_size = img.size
        img_resized = img.resize((224, 224)).convert('RGB')
        img_array = preprocess_input(np.expand_dims(img_to_array(img_resized), axis=0))

        # Get the top predicted class
        top_pred = np.argmax(global_model.predict(img_array))
        score = CategoricalScore([top_pred])

        # Ensure intensity is within bounds and scale it for steps
        intensity_bounded = max(1, min(100, intensity))
        steps = intensity_bounded * 10
        
        activation = activation_maximization(score, steps=steps)

        # Convert to NumPy array
        activation_arr = activation[0].numpy()
        activation_arr = normalize(activation_arr)
        
        # Create image and resize to original dimensions
        activation_img = Image.fromarray((activation_arr * 255).astype(np.uint8))
        activation_img = activation_img.resize(original_size, Image.LANCZOS)
        
        buffer = io.BytesIO()
        activation_img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
        
        print(f"[Activation Maximization] Generated visualization for class: {CLASS_INDEX[str(top_pred)][1]}")
        return image_data_url, CLASS_INDEX[str(top_pred)][1]

    except Exception as e:
        print(f"Error processing the image (Activation Maximization): {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return None, f"Error processing the image: {str(e)}"
