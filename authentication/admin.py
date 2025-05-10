from django.contrib import admin
from .models import UserProfile,CandidateTable,RecruiterTable
# # Register your models here.
admin.site.register(UserProfile)
admin.site.register(CandidateTable)
admin.site.register(RecruiterTable)