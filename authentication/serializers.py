from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .models import CandidateTable, User,RecruiterTable
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework import serializers


from rest_framework import serializers

class CandidateProfileSerializer(serializers.ModelSerializer):
    """Serializer for candidate profile data"""
    candidate_name = serializers.CharField(max_length=255)
    candidate_email = serializers.EmailField()
    candidate_cv = serializers.FileField(required=False, allow_null=True)  # Accepts file upload
    profile_image = serializers.ImageField(required=False, allow_null=True)  # Accepts image upload
    about = serializers.CharField(required=False, allow_blank=True)
    experience = serializers.CharField(required=False, allow_blank=True)
    skills = serializers.JSONField(required=False)
    education = serializers.CharField(required=False, allow_blank=True)
    professional_links = serializers.JSONField(required=False)
    location = serializers.CharField(required=False, allow_blank=True)
    bio = serializers.CharField(required=False, allow_blank=True)

    class Meta:
        model = CandidateTable
        exclude = ['candidate', 'id']


class RecruiterTableProfileSerializer(serializers.ModelSerializer):
    """Serializer for recruiter profile data"""
    recruiter_name = serializers.CharField(max_length=255)
    recruiter_email = serializers.EmailField()
    profile_image = serializers.ImageField(required=False, allow_null=True)  # Accepts image upload
    about = serializers.CharField(required=False, allow_blank=True)
    location = serializers.CharField(required=False, allow_blank=True)
    bio = serializers.CharField(required=False, allow_blank=True)
    recruiter_organization = serializers.CharField(max_length=255)
    company_information = serializers.JSONField(default=dict)
    
    class Meta:
        model = RecruiterTable
        exclude = ['recruiter', 'id']
