from rest_framework import serializers
from .models import JobPostingTable 
from .models import ApplicationTable
from authentication.models import CandidateTable
# class JobPostingSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = JobPostingTable
#         fields = '__all__'
from rest_framework import serializers
from .models import JobPostingTable, RecruiterTable

class JobPostingSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobPostingTable
        fields = ['id', 'title', 'description', 'skills', 'experience_level', 'location', 'status', 'created_at', 'is_private']
        read_only_fields = ['recruiter']

    def create(self, validated_data):
        user = self.context['request'].user
        try:
            recruiter = RecruiterTable.objects.get(recruiter=user)
        except RecruiterTable.DoesNotExist:
            # Create recruiter profile if it doesn't exist
            recruiter = RecruiterTable.objects.create(
                recruiter=user,
                recruiter_name=user.username  # Adjust fields as needed
            )
        
        job_posting = JobPostingTable.objects.create(
            recruiter=recruiter,
            **validated_data
        )
        return job_posting
from rest_framework import serializers
from .models import ApplicationTable

class ApplicationTableSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApplicationTable
        fields = '__all__'

class JobPostingWithApplicantsSerializer(serializers.ModelSerializer):
    applicants = serializers.SerializerMethodField()

    class Meta:
        model = JobPostingTable
        fields = ['job_title', 'skills', 'posted_at', 'applicants']

    def get_applicants(self, obj):
        applicants = ApplicationTable.objects.filter(job=obj)[:2]
        return [{'candidate_name': app.candidate.candidate_name} for app in applicants]
    
from rest_framework import serializers
from .models import ApplicationTable

class ApplicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApplicationTable
        
        fields = ['id', 'full_name', 'email', 'qualification', 'marks', 'resume', 'applied_at','application_status','interview_status']

class DetailedApplicationSerializer(serializers.ModelSerializer):
    job_title = serializers.CharField(source='job.title', read_only=True)

    class Meta:
        model = ApplicationTable
        fields = '__all__'  # Ensures all model fields are included

    # def get_resume_url(self, obj):
    #     return obj.resume.url if obj.resume else None
        # fields = [
        #     'id', 
        #     'full_name', 
        #     'email', 
        #     'qualification', 
        #     'marks', 
        #     'resume', 
        #     'applied_at',
        #     'job',
        #     'interview_status',
        #     'application_status',
        #     'job_title',  # Added this
        # ]

# Add this new serializer
class SimplifiedApplicantSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApplicationTable
        fields = ['id', 'full_name', 'marks']