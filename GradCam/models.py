import datetime

from django.db import models
from django.utils import timezone

# class that allows users to upload their images so they can be used in the program
class UserImage(models.Model):
    image = models.ImageField(upload_to="user_images/")
    upload_date = models.DateTimeField(auto_now_add=True)
    
'''
# we now need to tell the overall program that this app (gradcam) exists
# go to saliencyslider/settings.py and add "GradCam.apps.GradcamConfig" under INSTALLED_APPS
# we then execute $python manage.py makemigrations GradCam
# $python manage.py migrate  basically syncs up and applies any unapplied changes

# 3 Steps for making model changes:
    * Change your models (in models.py).
    * Run python manage.py makemigrations to create migrations for those changes
    * Run python manage.py migrate to apply those changes to the database.
'''









# # classes here subclass django.db.models.Model
# class Question(models.Model):
#     question_text = models.CharField(max_length=200) # some field classes have required arguments
#     pub_date = models.DateTimeField("date published") # you can pass a human-readable name as an argument

#     # now the question text is displayed when $Question.objects.all() is called
#     def __str__(self):
#         return self.question_text
    
#     # can replace this with my actual functions later
#     def was_published_recently(self):
#         return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


# class Choice(models.Model):
#     question = models.ForeignKey(Question, on_delete=models.CASCADE) # one-to-one, many-to-many or many-to-one relationships
#     choice_text = models.CharField(max_length=200)
#     votes = models.IntegerField(default=0) # some fields have optional arguments

#     def __str__(self):
#         return self.choice_text