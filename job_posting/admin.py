from django.contrib import admin
from .models import JobPostingTable,ApplicationTable

from django.contrib import admin
from .models import JobPostingTable

class JobPostingTableAdmin(admin.ModelAdmin):
    list_display = ['id', 'recruiter', 'title', 'description', 'skills', 'experience_level']
    # Update list or other filters that refer to 'user' to use 'recruiter'

admin.site.register(JobPostingTable, JobPostingTableAdmin)

admin.site.register(ApplicationTable)