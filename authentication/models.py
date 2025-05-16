from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # One-to-One relationship with User model
    is_recruiter = models.BooleanField(default=False)  # Custom field to mark if the user is a recruiter

    def __str__(self):
        return f"{self.user.username}'s Profile ({'Recruiter' if self.is_recruiter else 'Candidate'})"

from django.db import models
from django.contrib.auth.models import User
from cloudinary.models import CloudinaryField

class CandidateTable(models.Model):
    candidate = models.OneToOneField(
        User, 
        on_delete=models.CASCADE, 
        limit_choices_to={"userprofile__is_recruiter": False}
    )
    candidate_name = models.CharField(max_length=255)
    candidate_email = models.EmailField()
    candidate_cv = CloudinaryField('resume', blank=True, null=True)  # For storing CV/resume files
    profile_image = CloudinaryField('profile_image', blank=True, null=True)  # For storing profile images

    # profile_image = models.URLField(blank=True, null=True)  # Make optional
    about = models.TextField(blank=True, null=True)
    experience = models.TextField(blank=True, null=True)
    skills = models.JSONField(default=list)  # Default empty list
    education = models.TextField(blank=True, null=True)
    professional_links = models.JSONField(default=dict)  # Default empty dict
    resume = CloudinaryField('resume', blank=True, null=True)
    contact_information = models.JSONField(default=dict)  # Default empty dict
    location = models.TextField(blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    def save(self, *args, request_user=None, **kwargs):
        if self.pk:  # Editing an existing instance
            original = CandidateTable.objects.get(pk=self.pk)
            if (
                original.profile_image and
                original.profile_image != self.profile_image and
                (not request_user or not request_user.is_superuser)  # block non-admins
            ):
                raise ValueError("Profile image cannot be changed. Contact admin.")
        super().save(*args, **kwargs)


    def __str__(self):
        return self.candidate_name



class RecruiterTable(models.Model):
    recruiter = models.OneToOneField(User, on_delete=models.CASCADE, related_name="recruiter_profile")
    recruiter_organization = models.CharField(max_length=255)
    recruiter_name = models.CharField(max_length=255)
    location=models.TextField()
    bio=models.TextField() 
    about=models.TextField()
    company_information= models.JSONField()
    contact_information=models.JSONField()
    profile_image = CloudinaryField('profile_image', blank=True, null=True)  # For storing profile images

    def save(self, *args, **kwargs):
        if not self.recruiter.userprofile.is_recruiter:  # Check the UserProfile is marked as recruiter

            raise ValueError(f"User {self.user.username} is not marked as a recruiter.")
        super().save(*args, **kwargs)  # Call the original save method

    def __str__(self):
        return f"Recruiter: {self.recruiter_name} ({self.recruiter_organization})"
