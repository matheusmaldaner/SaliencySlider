import os
import io
import base64
import numpy as np
from PIL import Image
from django.conf import settings
from matplotlib import cm
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
        full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        img = Image.open(full_image_path).resize((224, 224)).convert('RGB')
        img_array = preprocess_input(np.expand_dims(img_to_array(img), axis=0))

        # Get the top predicted class
        top_pred = np.argmax(global_model.predict(img_array))
        score = CategoricalScore([top_pred])

        activation = activation_maximization(score, steps=intensity*10)

        # Convert to NumPy array
        activation_arr = activation[0].numpy()
        activation_arr = normalize(activation_arr)
        activation_img = Image.fromarray((activation_arr * 255).astype(np.uint8))
        
        buffer = io.BytesIO()
        activation_img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')
        return image_data_url, CLASS_INDEX[str(top_pred)][1]

    except Exception as e:
        print(f"Error processing the image (Activation Maximization): {e}")
        return None, "Error processing the image"
