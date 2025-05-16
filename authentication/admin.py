from django.contrib import admin
from .models import UserProfile,CandidateTable,RecruiterTable
# # Register your models here.
admin.site.register(UserProfile)
class CandidateTableAdmin(admin.ModelAdmin):
    def save_model(self, request, obj, form, change):
        # Pass the current user to  model's save method
        obj.save(request_user=request.user)

admin.site.register(CandidateTable, CandidateTableAdmin)
admin.site.register(RecruiterTable)