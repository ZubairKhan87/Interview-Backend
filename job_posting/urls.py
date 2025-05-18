from django.urls import path
from .views import JobPostingView,JobPostDetailView,JobPostListView,JobApplicationView,JobPostDetailPublic,JobPostingWithApplicantsView,ApplicantsView,ApplicantDetailView,CandidateAppliedJobsView,SortedApplicantsView,CurrentCandidateProfile,CandidatePendingInterview,CandidateInterviewDone,CandidateIncompleteInterview,RequestReInterview,RecruiterJobCompeletedInterviews,ReinterviewRecruiterRequests
from .views import get_application_status,download_frames_as_zip,export_interview_logs_csv,export_all_interview_logs_csv
urlpatterns = [
    path('jobposts/', JobPostListView.as_view(), name='job_post_list'),  # Correct URL pattern for the list view
    path('job_posting/', JobPostingView.as_view(), name='job_posting'),  # For post requests
    path('jobposts/<int:pk>/', JobPostDetailView.as_view(), name='jobpost-detail'),
    path('apply/', JobApplicationView.as_view(), name='apply'),
    path('publicposts/', JobPostDetailPublic.as_view(), name='jobpostspublic'),
    path('with_applicants/', JobPostingWithApplicantsView.as_view(), name='job_posts_with_applicants'),
    path('recruiter/completed/', RecruiterJobCompeletedInterviews.as_view(), name='recruiter-completed-interviews'),
    path('recruiter/reinterview/', ReinterviewRecruiterRequests.as_view(), name='recruiter-reinterviews'),
    path('job/<int:job_id>/applicants/', ApplicantsView.as_view(), name='job_applicants'),
    path('job/<int:job_id>/applicants/sorted/', SortedApplicantsView.as_view(), name='sorted-applicants'),
    path('applicant/<int:applicant_id>/', ApplicantDetailView.as_view(), name='applicant_details'),
    path('candidate/applied-jobs/', CandidateAppliedJobsView.as_view(), name='candidate-applied-jobs'),
    path('candidate/interview/pending/', CandidatePendingInterview.as_view(), name='candidate-pending-interviews'),
    path('candidate/interview/completed/', CandidateInterviewDone.as_view(), name='candidate-interview-jobs'),
    path('candidate/interview/incomplete/', CandidateIncompleteInterview.as_view(), name='candidate-incomplete-interviews'),
    path('candidate/interview/<int:interview_id>/request/', RequestReInterview.as_view(), name='request-interview'),
    path('application-status/<int:job_id>/', get_application_status, name='application-status'),
    path('current-candidate/profile/', CurrentCandidateProfile.as_view(), name='current-candidate'),
    path('download-frames/<int:application_id>/', download_frames_as_zip, name='download_frames'),
    path('export-interview-logs/<int:application_id>/', export_interview_logs_csv, name='export_interview_logs'),
    path('export-all-interview-logs/<int:job_id>/', export_all_interview_logs_csv, name='export_all_interview_logs'),




]