from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import UserProfile, CandidateTable, RecruiterTable

# # Create or update UserProfile
# @receiver(post_save, sender=User)
# def create_or_update_user_profile(sender, instance, created, **kwargs):
#     if created:
#         UserProfile.objects.create(user=instance)
#     else:
#         if not hasattr(instance, "userprofile"):
#             UserProfile.objects.create(user=instance)
#         else:
#             instance.userprofile.save()

# # Handle Recruiter creation or removal
# @receiver(post_save, sender=UserProfile)
# def create_or_update_recruiter(sender, instance, created, **kwargs):
#     if instance.is_recruiter:
#         RecruiterTable.objects.update_or_create(
#             recruiter=instance.user,
#             defaults={
#                 "recruiter_name": instance.user.username,
#                 "recruiter_organization": "Default Organization",
#             },
#         )
#     else:
#         RecruiterTable.objects.filter(recruiter=instance.user).delete()

# # Handle Candidate creation
# @receiver(post_save, sender=UserProfile)
# def create_or_update_candidate(sender, instance, created, **kwargs):
#     if not instance.is_recruiter:
#         CandidateTable.objects.update_or_create(
#             candidate=instance.user,
#             defaults={
#                 "candidate_name": instance.user.username,
#                 "candidate_email": instance.user.email,
#             },
#         )
#     else:
#         CandidateTable.objects.filter(candidate=instance.user).delete()



@receiver(post_save, sender=UserProfile)
def create_or_update_role_tables(sender, instance, created, **kwargs):
    # Ensure exclusivity before proceeding
    if instance.is_recruiter:
        # Delete candidate if exists
        CandidateTable.objects.filter(candidate=instance.user).delete()
        # Create or update recruiter
        RecruiterTable.objects.update_or_create(
            recruiter=instance.user,
            defaults={
                "recruiter_name": instance.user.username,
                "recruiter_organization": "Default Organization",
                "location": "",
                "bio": "",
                "about": "",
                "company_information": {},
                "contact_information": {},
            },
        )
    else:
        # Delete recruiter if exists
        RecruiterTable.objects.filter(recruiter=instance.user).delete()
        # Create or update candidate
        CandidateTable.objects.update_or_create(
            candidate=instance.user,
            defaults={
                "candidate_name": instance.user.username,
                "candidate_email": instance.user.email,
            },
        )
