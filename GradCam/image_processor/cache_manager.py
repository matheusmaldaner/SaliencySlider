import os
import json
import time
import threading
import numpy as np
from PIL import Image
from pathlib import Path
import base64
import io
from django.conf import settings
import traceback

from .gradcam_processor import process_image_gradcam
from .gradcamPP_processor import process_image_gradcamPP
from .scorecam_processor import process_image_scorecam
from .vanilla_gradients_processor import process_image_vanilla_gradients
from .integrated_gradients_processor import process_image_integrated_gradients
from .activation_maximization_processor import process_image_activation_maximization

# Dictionary to store cached results
# Structure: {image_path: {method: {intensity: (data_url, class_label)}}}
saliency_cache = {}

# List of methods available
METHODS = [
    "gradcam",
    "gradcamPP",
    "score-cam",
    "vanilla-gradients",
    "integrated-gradients",
    "activation-maximization"
]

def get_processor_func(method):
    """Return the appropriate processor function for a given method."""
    if method == "gradcam":
        return process_image_gradcam
    elif method == "gradcamPP":
        return process_image_gradcamPP
    elif method == "score-cam":
        return process_image_scorecam
    elif method == "vanilla-gradients":
        return process_image_vanilla_gradients
    elif method == "integrated-gradients":
        return process_image_integrated_gradients
    elif method == "activation-maximization":
        return process_image_activation_maximization
    else:
        raise ValueError(f"Unknown method: {method}")

def get_cache_path():
    """Get the path to the JSON file for persisting cache."""
    cache_dir = os.path.join(settings.MEDIA_ROOT, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, 'saliency_cache.json')

def save_cache():
    """Save the current cache to a JSON file."""
    try:
        # Convert to a serializable format - data URLs can be very long,
        # so we'll save only the image paths and methods that are cached
        serializable_cache = {}
        for img_path, methods in saliency_cache.items():
            serializable_cache[img_path] = list(methods.keys())
        
        with open(get_cache_path(), 'w') as f:
            json.dump(serializable_cache, f)
        print("Saved cache index to disk")
    except Exception as e:
        print(f"Error saving cache: {e}")

def load_cached_result(image_path, method, intensity):
    """
    Load a cached result if available.
    Returns (image_data_url, class_label) or (None, None) if not cached.
    """
    if image_path in saliency_cache and method in saliency_cache[image_path]:
        # If exact intensity is cached, return it
        if intensity in saliency_cache[image_path][method]:
            return saliency_cache[image_path][method][intensity]
    return None, None

def save_to_cache(image_path, method, intensity, image_data_url, class_label):
    """Save a result to the in-memory cache."""
    if image_path not in saliency_cache:
        saliency_cache[image_path] = {}
    
    if method not in saliency_cache[image_path]:
        saliency_cache[image_path][method] = {}
    
    saliency_cache[image_path][method][intensity] = (image_data_url, class_label)

def is_demo_image(image_path):
    """Check if the image is one of our precomputed demo images."""
    demo_dir = 'demo_images'
    return demo_dir in image_path

def precompute_saliency_maps(demo_images, intensities=None, methods=None):
    """
    Precompute saliency maps for demo images.
    
    Args:
        demo_images: List of relative paths to demo images
        intensities: List of intensity values to compute (default: 0-100 in increments of 5)
        methods: List of methods to compute (default: all methods)
    """
    if intensities is None:
        intensities = list(range(0, 101, 5))  # 0, 5, 10, ..., 100
    
    if methods is None:
        methods = METHODS
    
    print(f"Starting precomputation for {len(demo_images)} images, {len(methods)} methods, {len(intensities)} intensities...")
    
    # Track progress
    total_combinations = len(demo_images) * len(methods) * len(intensities)
    processed = 0
    start_time = time.time()
    
    for image_path in demo_images:
        for method in methods:
            processor_func = get_processor_func(method)
            
            # Process first intensity to get the class label
            intensity = intensities[0]
            print(f"Processing {image_path} with {method} at intensity {intensity}")
            try:
                image_data_url, class_label = processor_func(image_path, intensity)
                if image_data_url:
                    save_to_cache(image_path, method, intensity, image_data_url, class_label)
                    processed += 1
                    
                    # Process remaining intensities
                    for intensity in intensities[1:]:
                        image_data_url, _ = processor_func(image_path, intensity)
                        if image_data_url:
                            save_to_cache(image_path, method, intensity, image_data_url, class_label)
                        processed += 1
                        
                        # Show progress
                        if processed % 10 == 0:
                            elapsed = time.time() - start_time
                            progress = processed / total_combinations * 100
                            print(f"Progress: {progress:.1f}% ({processed}/{total_combinations}) - Elapsed time: {elapsed:.1f}s")
            except Exception as e:
                print(f"Error processing {image_path} with {method}: {e}")
                print(traceback.format_exc())
    
    # Save cache metadata after all processing is done
    save_cache()
    print(f"Precomputation completed in {time.time() - start_time:.1f}s")
    return processed

def start_precomputation_thread(demo_images):
    """Start precomputation in a background thread."""
    thread = threading.Thread(
        target=precompute_saliency_maps,
        args=(demo_images,),
        daemon=True
    )
    thread.start()
    return thread

def get_saliency_result(image_path, method, intensity):
    """
    Get saliency result for an image, using cache if available.
    
    Args:
        image_path: Path to the image
        method: Saliency method to use
        intensity: Intensity value (0-100)
        
    Returns:
        (image_data_url, class_label) tuple
    """
    # Check cache first
    image_data_url, class_label = load_cached_result(image_path, method, intensity)
    if image_data_url:
        print(f"Cache hit for {image_path}, {method}, intensity {intensity}")
        return image_data_url, class_label
    
    # Not in cache, compute it
    processor_func = get_processor_func(method)
    image_data_url, class_label = processor_func(image_path, intensity)
    
    # If it's a demo image, cache the result
    if is_demo_image(image_path) and image_data_url:
        save_to_cache(image_path, method, intensity, image_data_url, class_label)
        
    return image_data_url, class_label 