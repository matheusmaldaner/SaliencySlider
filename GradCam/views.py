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

from .models import UserImage
from .forms import UserImageForm
import os

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

def update_image(request):
    try:
        if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            data = json.loads(request.body)
            intensity = data.get('intensity', 0)
            image_path = data.get('image_path', '')
            method = data.get('method', 'gradcam')  # default to gradcam if not provided

            if image_path.startswith('http://0.0.0.0:8000/media/'):
                image_path = image_path.replace('http://0.0.0.0:8000/media/', '')

            if image_path.startswith('http://localhost:8000/media/'):
                image_path = image_path.replace('http://localhost:8000/media/', '')

            if not image_path:
                return JsonResponse({'error': 'Image path is empty'}, status=400)

            print(f"Processing image: {image_path}, with intensity: {intensity}, method: {method}")

            if method == "gradcam":
                image_data_url, highest_pred_label = process_image_gradcam(image_path, int(intensity))
            elif method == "activation-maximization":
                image_data_url, highest_pred_label = process_image_activation_maximization(image_path, int(intensity))
            elif method == "integrated-gradients":
                image_data_url, highest_pred_label = process_image_integrated_gradients(image_path, int(intensity))
            elif method == "vanilla-gradients":
                image_data_url, highest_pred_label = process_image_vanilla_gradients(image_path, int(intensity))
            elif method == "score-cam":
                image_data_url, highest_pred_label = process_image_scorecam(image_path, int(intensity))
            elif method == "gradcamPP":
                image_data_url, highest_pred_label = process_image_gradcamPP(image_path, int(intensity))
            else:
                return JsonResponse({'error': 'Invalid method'}, status=400)

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

