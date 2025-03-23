from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

# in case Django has many apps, this differentiates them
app_name = "GradCam"

urlpatterns = [
    path("", views.gradcam_landing, name="gradcam_landing"),
    path("home", views.home, name="gradcam_home"),
    path("GradCam/", views.index, name="index"),
    path("display_image/", views.display_image, name="display_image"),
    path("user_image/", views.user_image, name="user_image"),
    path("success/", views.success, name='success'),
    path('raw_images/', views.display_raw_images, name = 'raw_images'),
    path('last_image/', views.display_last_image, name="last_image"),
    path('update_image/', views.update_image, name='update_image'),
    
    # Demo images routes
    path('demo/', views.demo_images, name='demo_images'),
    path('demo/<str:filename>/', views.view_demo_image, name='view_demo_image'),
    
    # Grid view of all saliency models
    path('all_models/<str:filename>/', views.all_models_view, name='all_models_view'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


