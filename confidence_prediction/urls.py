from django.urls import path
from .views import FaceVerificationView, FaceVerificationCheat,analyze_confidence
from .views import analyze_confidence

urlpatterns = [
    path('verify-face/', FaceVerificationView.as_view(), name='verify_face'),
    path('verify-face/cheat/', FaceVerificationCheat.as_view(), name='verify_face'),
    path('analyze-confidence/', analyze_confidence, name='analyze-confidence'),
]
