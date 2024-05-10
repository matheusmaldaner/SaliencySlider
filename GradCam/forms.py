from django import forms
from django.core.exceptions import ValidationError
from .models import UserImage
import os

class UserImageForm(forms.ModelForm):

    class Meta:
        model = UserImage
        fields = ['image']

    def clean_image(self):
        image = self.cleaned_data.get('image')
        if image:
            # checks the file extension
            ext = os.path.splitext(image.name)[1].lower()  # Get the file extension
            valid_extensions = ['.png', '.jpg', '.jpeg']
            if not ext.lower() in valid_extensions:
                raise ValidationError('Unsupported file extension.')
        return image