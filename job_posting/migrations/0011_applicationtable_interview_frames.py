# Generated by Django 5.1.4 on 2025-02-14 10:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('job_posting', '0010_alter_applicationtable_unique_together'),
    ]

    operations = [
        migrations.AddField(
            model_name='applicationtable',
            name='interview_frames',
            field=models.URLField(default=1),
            preserve_default=False,
        ),
    ]
