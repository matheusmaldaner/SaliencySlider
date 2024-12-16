from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

# in case Django has many apps, this differentiates them
app_name = "GradCam"

urlpatterns = [
    path("", views.home, name="home"),
    path("GradCam/", views.index, name="index"),
    path("display_image/", views.display_image, name="display_image"),
    path("user_image/", views.user_image, name="user_image"),
    path("success/", views.success, name='success'),
    path('raw_images/', views.display_raw_images, name = 'raw_images'),
    path('last_image/', views.display_last_image, name="last_image"),
    path('update_image/', views.update_image, name='update_image'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
