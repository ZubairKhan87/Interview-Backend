from django.urls import path
from . import views
from .views import chatbot_page, chatbot_response,get_current_candidate,save_frame,get_csrf_token,transcribe_audio
urlpatterns = [
    path('', views.chatbot_response, name='chatbot_response'),  # Match '/chat/' to a view
    path('chatbot/', chatbot_page, name='chatbot_page'),
    path('chat/', chatbot_response, name='chatbot_response'),
    path('current-candidate/', get_current_candidate, name='current-candidate'),
    path('save-frame/', save_frame, name='save-frame'),
    path('get-csrf-token/', get_csrf_token, name='get-csrf-token'),
    path('transcribe/', transcribe_audio, name='transcribe-audio'),

]

