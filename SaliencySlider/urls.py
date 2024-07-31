from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static 

urlpatterns = [
    # you should always use "include", except with admin/
    path("", include("GradCam.urls")), # this is the root URL
    path("GradCam/", include("GradCam.urls")),
    
    path("admin/", admin.site.urls),
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, 
                          document_root=settings.MEDIA_ROOT)
