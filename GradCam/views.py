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
from .image_processor import process_image

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

            if not image_path:
                return JsonResponse({'error': 'Image path is empty'}, status=400)

            image_data_url, highest_pred_label = process_image(image_path, int(intensity))
            return JsonResponse({
                'image_data_url': image_data_url,      # return the modified image
                'predicted_class': highest_pred_label  # return the predicted class
            })
        else:
            return JsonResponse({'error': 'Invalid request'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
        
# # Create your views here.
# def index(request):
#     latest_question_list = Question.objects.order_by("-pub_date")[:5]
#     # now we can use the html template we created
#     context = {"latest_question_list": latest_question_list}
#     return render(request, "GradCam/index.html", context) # returns HttpResponse object

# # slightly different because it takes an argument
# def detail(request, question_id):
#     # raises Http404 if object is not found
#     question = get_object_or_404(Question, pk=question_id) # could be get_list_or_404 to check for empty list
#     return render(request, "GradCam/detail.html", {"question": question})

# def results(request, question_id):
#     response = "You're looking at the results of question %s."
#     return HttpResponse(response % question_id)

# def vote(request, question_id):
#     question = get_object_or_404(Question, pk=question_id)
#     try:
#         selected_choice = question.choice_set.get(pk=request.POST["choice"]) # request.POST values r always strings
#     except (KeyError, Choice.DoesNotExist):
#         # Redisplay the question voting form.
#         return render(
#             request,
#             "GradCam/detail.html",
#             {
#                 "question": question,
#                 "error_message": "You didn't select a choice.",
#             },
#         )
#     else:
#         selected_choice.votes = F("votes") + 1
#         selected_choice.save()
        
#         # Always return an HttpResponseRedirect after successfully dealing
#         # with POST data. This prevents data from being posted twice if a
#         # user hits the Back button.
#         return HttpResponseRedirect(reverse("GardCam:results", args=(question.id,)))  

