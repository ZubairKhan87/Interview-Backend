# Generated by Django 5.1.3 on 2024-12-14 21:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('job_posting', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='applicationtable',
            name='email',
            field=models.EmailField(default=1, max_length=254),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='applicationtable',
            name='full_name',
            field=models.CharField(default=1, max_length=110),
            preserve_default=False,
        ),
    ]
