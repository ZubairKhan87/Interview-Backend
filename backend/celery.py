import os
from celery import Celery

# Set default Django settings module for 'celery'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")  # Change if your Django project folder is named differently

app = Celery("backend")  # Match with your Django project folder
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
