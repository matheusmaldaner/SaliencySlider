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

# initializes global model
global_model = VGG19(weights='imagenet')
global_model.layers[-1].activation = tf.keras.activations.linear
model = utils.apply_modifications(global_model)

# loads class index
url = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
response = requests.get(url)
CLASS_INDEX = response.json()

# initializes Gradcam with modified model
gradcam = Gradcam(model, clone=False)

def process_image(image_url, intensity):

    # fetches image from URL
    response = requests.get(image_url)  

    # saves unaltered copy for display
    img_copy = Image.open(io.BytesIO(response.content))  
    img_copy = img_copy.resize((224, 224))

    img = Image.open(io.BytesIO(response.content))  # opens image from a bytes buffer
    # convert it to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
        img_copy = img_copy.convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)

    
    img_array = np.expand_dims(img_array, axis=0)  # Make 'batch' of 1
    img_array = preprocess_input(img_array)

    # Predictions
    predictions = global_model.predict(img_array) 
    top_pred = np.argmax(predictions[0]) # can change to retrieve more predictions like 1-5
    
    classlabel = []
    for i_dict in range(len(CLASS_INDEX)):
        classlabel.append(CLASS_INDEX[str(i_dict)][1])

    # predicts top X classes predicted
    class_idxs_sorted1 = np.argsort(predictions.flatten())[::-1]
    X = 5

    # can modify this part to retrieve X number of top classes rather than just the top pred
    for i, idx in enumerate(class_idxs_sorted1[:X]):
        print("Top {} predicted class:     Pr(Class={:18} [index={}])={:5.3f}".format(
            + 1,classlabel[idx],idx,predictions[0,idx]))
    
        # stores index and classification of highest prediction
        if i == 0:
            highest_pred_idx = idx
            highest_pred_label = classlabel[idx]    
    
    # GradCam stuff -- might be beneficial to take it out of this function somehow
    score = CategoricalScore([highest_pred_idx])
    input_images = preprocess_input(img_array)

    # generates heatmap with GradCAM and resizes to match image dims
    cam = gradcam(score, input_images, penultimate_layer=-1)
    # heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    # heatmap_img = Image.fromarray(heatmap)
    # heatmap_img = heatmap_img.resize(img.size, Image.LANCZOS).convert('RGBA')

    # X percent of features the model considers the most important
    PERCENTAGE_IMPORTANCE = intensity
    threshold_value = np.percentile(cam[0], 100-PERCENTAGE_IMPORTANCE)

    # 2d and 3d masks
    mask = cam[0] > threshold_value
    mask3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # applies mask to original image to show important regions
    importance_img = img_copy * mask3d
    importance_img = Image.fromarray(importance_img)
    if importance_img.mode != 'RGB':
        importance_img = importance_img.convert('RGB')    

    # converts the processed image to a data URL
    buffer = io.BytesIO()
    importance_img.save(buffer, format="JPEG")  # Save image to the buffer in JPEG format
    image_data = buffer.getvalue()
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_data).decode('utf-8')

    return image_data_url, highest_pred_label



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