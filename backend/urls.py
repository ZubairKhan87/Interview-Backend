"""
URL configuration for NLU project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings  # Add this import

from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static  # Add this import

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # path('', include('ner_app.urls')),
    # path('api/', include('Emotion_Detection.urls')),
    # path('api/question_management/', include('question_management.urls')),  # Add this line
    path('api/authentication/', include('authentication.urls')),  # Add this line
    path('api/job_posting/', include('job_posting.urls')),
    path('api/chat/', include('interview.urls')),  # Include URLs from the chatbot app
    # path('api/confidence_prediction/', include('confidence_prediction.urls')),  # Add this line
    path('auth/', include('social_django.urls', namespace='social')),  # Adds Google OAuth endpoints


]
# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

