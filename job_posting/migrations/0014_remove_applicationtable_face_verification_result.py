# Generated by Django 5.1.4 on 2025-02-21 07:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('job_posting', '0013_applicationtable_confidence_score_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='applicationtable',
            name='face_verification_result',
        ),
    ]
