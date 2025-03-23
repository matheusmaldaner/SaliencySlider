import json
from django.db.models import F
from django.shortcuts import get_object_or_404, render, redirect
from django.http import Http404 # to raise 404 errors
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader # eg. template = loader.get_template("GradCam/index.html")
from django.urls import reverse
from django import forms
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import re
import os
from urllib.parse import urlparse

from .models import UserImage
from .forms import UserImageForm

from .image_processor.cache_manager import get_saliency_result, start_precomputation_thread

# Import all processors for backward compatibility
from .image_processor.gradcam_processor import process_image_gradcam
from .image_processor.activation_maximization_processor import process_image_activation_maximization
from .image_processor.integrated_gradients_processor import process_image_integrated_gradients
from .image_processor.vanilla_gradients_processor import process_image_vanilla_gradients
from .image_processor.scorecam_processor import process_image_scorecam
from .image_processor.gradcamPP_processor import process_image_gradcamPP


def home(request):
    return render(request, 'GradCam/gradcam_home.html')

def gradcam_landing(request):
    return render(request, 'GradCam/gradcam_landing.html')

def index(request):
    return render(request, 'GradCam/index.html')

# test function to display a placeholder image
def display_image(request):
    # find a way to update this dynamically
    image_path = "apple.jpg"
    with open(image_path, 'rb') as image:
        return HttpResponse(image.read(), content_type="image/png")
    

def user_image(request):
    # check if request is POST -- data is submitted
    if request.method == 'POST':
        form = UserImageForm(request.POST, request.FILES)
        
        # checks if all required fields in the form are filled
        if form.is_valid():
            form.save() # saves the form data into database
            return redirect('GradCam:last_image')
        
    # if form is invalid, it just retries it
    else:
        form = UserImageForm()

    return render(request, 'GradCam/index.html', {'form': form})   
    
def success(request):
    return HttpResponse('successfully uploaded')
    
def display_raw_images(request):
    if request.method == 'GET':

        UserImages = UserImage.objects.all()
        return render(request, 'GradCam/display_raw_images.html',
                       {'raw_images': UserImages})

def display_last_image(request):
    if request.method == 'GET':
        # '-upload_date' orders descending, 'first()' gets the most recent
        last_image = UserImage.objects.order_by('-upload_date').first()  
        return render(request, 'GradCam/display_last_image.html', {'last_image': last_image})

def demo_images(request):
    """Display a gallery of demo images with precomputed saliency maps"""
    # List of demo images (relative paths) to display
    from django.conf import settings
    
    # Create demo_images directory if it doesn't exist
    demo_dir = os.path.join(settings.MEDIA_ROOT, 'demo_images')
    os.makedirs(demo_dir, exist_ok=True)
    
    # List demo images (if any exist)
    demo_images = []
    if os.path.exists(demo_dir):
        for filename in os.listdir(demo_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join('demo_images', filename)
                demo_images.append({
                    'name': filename,
                    'path': image_path,
                    'url': os.path.join(settings.MEDIA_URL, image_path)
                })
    
    # Removed automatic precomputation to prevent navigation lock
    
    context = {
        'demo_images': demo_images,
        'media_url': settings.MEDIA_URL
    }
    
    return render(request, 'GradCam/demo_images.html', context)

def view_demo_image(request, filename):
    """Display a single demo image with precomputed saliency maps"""
    image_path = os.path.join('demo_images', filename)
    
    # Create a simple object similar to UserImage to reuse the template
    class DemoImage:
        def __init__(self, path):
            self.image = path
            self.url = os.path.join('/media/', path)
    
    # We're reusing the display_last_image template
    return render(request, 'GradCam/display_last_image.html', 
                  {'last_image': DemoImage(image_path)})

def extract_media_path(url_path):
    """
    Extract the media path from a URL, regardless of protocol or domain.
    For example, 'http://127.0.0.1:8000/media/user_images/image.jpg' becomes 'user_images/image.jpg'
    """
    # Check if it's already just a relative path
    if not url_path.startswith(('http://', 'https://')):
        return url_path
    
    # Parse the URL to extract the path component
    parsed_url = urlparse(url_path)
    path = parsed_url.path
    
    # Find the 'media/' part and extract everything after it
    match = re.search(r'/media/(.*?)$', path)
    if match:
        return match.group(1)
    
    return url_path  # Return the original if we can't extract

def update_image(request):
    try:
        if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            data = json.loads(request.body)
            intensity = data.get('intensity', 0)
            image_path = data.get('image_path', '')
            method = data.get('method', 'gradcam')  # default to gradcam if not provided

            # Extract the media path from the URL, regardless of domain or protocol
            image_path = extract_media_path(image_path)

            if not image_path:
                return JsonResponse({'error': 'Image path is empty'}, status=400)

            print(f"Processing image: {image_path}, with intensity: {intensity}, method: {method}")
            
            # Use cache manager to get the result
            image_data_url, highest_pred_label = get_saliency_result(image_path, method, int(intensity))

            if not image_data_url:
                return JsonResponse({'error': 'Image processing failed'}, status=500)

            print(f"Processed image, predicted class: {highest_pred_label}")
            return JsonResponse({
                'image_data_url': image_data_url,
                'predicted_class': highest_pred_label
            })
        else:
            return JsonResponse({'error': 'Invalid request'}, status=400)
    except Exception as e:
        print(f"Error processing image: {e}")
        return JsonResponse({'error': str(e)}, status=500)

def all_models_view(request, filename):
    """Display all 6 saliency models in a 3x2 grid for a single image"""
    image_path = os.path.join('demo_images', filename)
    
    # Get intensity from the request or use default
    intensity = int(request.GET.get('intensity', 50))
    
    # Create a simple object similar to UserImage to reuse template logic
    class DemoImage:
        def __init__(self, path):
            self.image = path
            self.url = os.path.join('/media/', path)
    
    demo_image = DemoImage(image_path)
    
    # Define all available methods - use the actual method names from cache_manager.py
    methods = ['gradcam', 'gradcamPP', 'score-cam', 'vanilla-gradients', 'integrated-gradients', 'activation-maximization']
    
    # Compute results for all methods at the specified intensity
    results = {}
    model_predictions = {}
    processed_count = 0
    total_methods = len(methods)
    
    # Process all methods
    for method in methods:
        processed_count += 1
        print(f"Processing {method} ({processed_count}/{total_methods})...")
        
        try:
            image_data_url, prediction = get_saliency_result(image_path, method, intensity)
            results[method] = image_data_url
            model_predictions[method] = prediction
        except Exception as e:
            print(f"Error processing {method}: {e}")
            # Use the original image as a fallback
            results[method] = demo_image.url
            model_predictions[method] = f"Error: Processing failed"
    
    # Organize methods into a grid layout (2x3)
    grid = [
        ['gradcam', 'gradcamPP', 'score-cam'],
        ['vanilla-gradients', 'integrated-gradients', 'activation-maximization']
    ]
    
    context = {
        'image': demo_image,
        'results': results,
        'predictions': model_predictions,
        'methods': methods,
        'grid': grid,
        'intensity': intensity,
        'processed_count': processed_count,
        'total_methods': total_methods
    }
    
    return render(request, 'GradCam/all_models_grid.html', context)

