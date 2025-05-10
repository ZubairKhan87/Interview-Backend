from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import JobPostingTable
from .serializers import JobPostingSerializer
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .models import ApplicationTable
from rest_framework.permissions import IsAuthenticated,AllowAny
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics,permissions
from .models import JobPostingTable
from .serializers import JobPostingSerializer,ApplicationSerializer,DetailedApplicationSerializer,SimplifiedApplicantSerializer
from authentication.models import RecruiterTable
from django.core.exceptions import PermissionDenied
from rest_framework.decorators import api_view, permission_classes
from django.conf import settings
import requests



#Linkedin Authorization And Job Posting on Linkedin
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import requests
import json
import urllib.parse
from base64 import b64encode

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import requests
import urllib.parse
import secrets
import json
import logging
# views.py
# views.py
logger = logging.getLogger(__name__)

# views.py
import logging
import secrets
import urllib.parse
import requests
import hashlib
import base64
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings

logger = logging.getLogger(__name__)

def generate_code_verifier():
    code_verifier = secrets.token_urlsafe(64)
    return code_verifier

def generate_code_challenge(code_verifier):
    code_challenge = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode('utf-8')
    code_challenge = code_challenge.replace('=', '')
    return code_challenge

class LinkedInAuthURLView(APIView):
    authentication_classes = []
    permission_classes = []
    
    def get(self, request):
        try:
            # Generate PKCE values
            code_verifier = generate_code_verifier()
            code_challenge = generate_code_challenge(code_verifier)
            
            # Generate state
            state = secrets.token_urlsafe(32)
            
            # Store in session
            request.session['linkedin_code_verifier'] = code_verifier
            request.session['linkedin_state'] = state
            request.session.modified = True  # Force session save
            
            logger.info(f"Stored in session - verifier: {bool(code_verifier)}, state: {bool(state)}")
            
            scopes = [
                'openid',
                'profile',
                'email',
                'w_member_social'
            ]
            
            auth_params = {
                'response_type': 'code',
                'client_id': settings.LINKEDIN_CLIENT_ID,
                'redirect_uri': settings.LINKEDIN_REDIRECT_URI,
                'scope': ' '.join(scopes),
                'state': state,
                'code_challenge': code_challenge,
                'code_challenge_method': 'S256'
            }
            
            auth_url = 'https://www.linkedin.com/oauth/v2/authorization?' + \
                      urllib.parse.urlencode(auth_params)
            
            logger.info(f"Generated LinkedIn auth URL with params: {auth_params}")
            
            return Response({
                'auth_url': auth_url,
                'state': state,
                'code_verifier': code_verifier  # Send this back to frontend
            })
            
        except Exception as e:
            logger.error(f"Error in auth URL view: {str(e)}")
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class LinkedInShareView(APIView):
    authentication_classes = []
    permission_classes = []
    
    def post(self, request):
        try:
            code = request.data.get('code')
            job_data = request.data.get('jobData')
            code_verifier = request.data.get('code_verifier')
            
            logger.info("=== Starting LinkedIn Share Process ===")
            logger.info(f"Authorization code present: {bool(code)}")
            logger.info(f"Job data present: {bool(job_data)}")
            logger.info(f"Code verifier present: {bool(code_verifier)}")
            
            if not all([code, job_data, code_verifier]):
                missing = []
                if not code: missing.append('code')
                if not job_data: missing.append('job_data')
                if not code_verifier: missing.append('code_verifier')
                return Response(
                    {'error': f'Missing required data: {", ".join(missing)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Token exchange endpoint
            token_url = 'https://www.linkedin.com/oauth/v2/accessToken'
            
            # Include all parameters in the request body
            token_params = {
                'grant_type': 'authorization_code',
                'code': code,
                'redirect_uri': settings.LINKEDIN_REDIRECT_URI,
                'client_id': settings.LINKEDIN_CLIENT_ID,
                'client_secret': settings.LINKEDIN_CLIENT_SECRET,
                'code_verifier': code_verifier
            }
            
            # Basic auth header (as backup)
            auth_str = f"{settings.LINKEDIN_CLIENT_ID}:{settings.LINKEDIN_CLIENT_SECRET}"
            auth_bytes = auth_str.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                'Authorization': f'Basic {auth_b64}',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            logger.info("=== Token Request Details ===")
            logger.info(f"Token URL: {token_url}")
            logger.info(f"Redirect URI: {settings.LINKEDIN_REDIRECT_URI}")
            logger.info(f"Client ID present in params: {bool(token_params.get('client_id'))}")
            logger.info(f"Auth header present: {bool(headers.get('Authorization'))}")
            
            # Log the exact parameters being sent (excluding sensitive data)
            safe_params = token_params.copy()
            safe_params['client_secret'] = '***'
            safe_params['code'] = '***'
            logger.info(f"Token parameters: {safe_params}")
            
            # Make token request
            logger.info("Making token request...")
            token_response = requests.post(
                token_url, 
                data=token_params,  # Send params as form data
                headers=headers,
                verify=True
            )
            
            logger.info(f"Token response status code: {token_response.status_code}")
            logger.info(f"Token response headers: {dict(token_response.headers)}")
            
            # Handle token response
            if token_response.status_code != 200:
                error_data = {}
                try:
                    error_data = token_response.json()
                except Exception as e:
                    logger.error(f"Failed to parse error response: {str(e)}")
                    error_data = {'text': token_response.text}
                
                logger.error(f"Token error response: {error_data}")
                
                # Enhanced error messages
                if error_data.get('error') == 'invalid_request':
                    details = error_data.get('error_description', 'Missing or invalid parameters')
                    logger.error(f"Invalid request error: {details}")
                    return Response({
                        'error': 'Invalid request parameters',
                        'details': details,
                        'sent_params': list(safe_params.keys())  # Log what params were sent
                    }, status=status.HTTP_400_BAD_REQUEST)
                elif error_data.get('error') == 'invalid_client':
                    return Response({
                        'error': 'Invalid client credentials',
                        'details': 'Please verify your LinkedIn client ID and secret'
                    }, status=status.HTTP_401_UNAUTHORIZED)
                
                return Response({
                    'error': 'Failed to obtain access token',
                    'details': error_data.get('error_description', 'Unknown error occurred')
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Parse successful response
            token_data = token_response.json()
            access_token = token_data.get('access_token')
            
            if not access_token:
                logger.error("No access token in successful response")
                return Response({
                    'error': 'No access token received',
                    'details': token_data
                }, status=status.HTTP_400_BAD_REQUEST)
            
            logger.info("Successfully obtained access token")
            
            # Return success response
            return Response({
                'status': 'Successfully authenticated',
                'token_type': token_data.get('token_type'),
                'expires_in': token_data.get('expires_in')
            })
            
        except Exception as e:
            logger.error(f"Unexpected error in share view: {str(e)}", exc_info=True)
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # 
# My Application
class JobPostingView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = JobPostingSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            job = serializer.save()
            
            # Add the job URL to the response
            response_data = serializer.data
            response_data['share_url'] = f"{settings.FRONTEND_URL}/jobs/{job.id}/apply"
            
            return Response(response_data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def get(self, request):
        posts = JobPostingTable.objects.all().order_by('-created_at')  # Add this method
        serializer = JobPostingSerializer(posts, many=True)
        return Response(serializer.data)
class JobPostListView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = JobPostingSerializer
    
    def get_queryset(self):
        user = self.request.user
        
        # If user is a recruiter, show only their job posts
        if hasattr(user, 'recruiter_profile'):
            recruiter = user.recruiter_profile
            return JobPostingTable.objects.filter(recruiter=recruiter)
        
        # If user is a candidate, show all job posts
        return JobPostingTable.objects.all()

class JobPostDetailView(generics.RetrieveUpdateDestroyAPIView):
    permission_classes = [IsAuthenticated]
    queryset = JobPostingTable.objects.all()
    serializer_class = JobPostingSerializer

    def perform_destroy(self, instance):
        # Ensure only the job's recruiter can delete
        user = self.request.user
        recruiter = RecruiterTable.objects.filter(recruiter_id=user).first()
        
        if instance.recruiter != recruiter:
            raise PermissionDenied("You are not authorized to delete this job posting.")
        
        instance.delete()


# Public posts and Job filtering + applying Search Functionality

class JobPostDetailPublic(generics.ListAPIView):
    
    # queryset = JobPostingTable.objects.filter(is_private=False)  # Filter to exclude private jobs
    serializer_class = JobPostingSerializer
    permission_classes = [AllowAny]  # This allows unauthenticated access
    def get_queryset(self):
        user = self.request.user
        print('user is ',user)
        queryset = JobPostingTable.objects.filter(is_private=False)  # Exclude private jobs

        if user.is_authenticated:
            # Get jobs the candidate has already applied for
            applied_jobs = ApplicationTable.objects.filter(candidate__candidate=user).values_list('job', flat=True)
            # Exclude those jobs from the queryset
            queryset = queryset.exclude(id__in=applied_jobs)
        print("queryset",queryset)
        return queryset
    # permission_classes = [permissions.IsAuthenticatedOrReadOnly]  # Custom permissions can also be added

# job_posting/views.py
class JobPostingWithApplicantsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            user = request.user
            # Try to get or create recruiter profile
            recruiter, created = RecruiterTable.objects.get_or_create(
                recruiter=user,
                defaults={'recruiter_name': user.username}  # Adjust fields as needed
            )
            
            job_posts = JobPostingTable.objects.filter(recruiter=recruiter).order_by('-created_at')

            result = []
            for job in job_posts:
                applicants = ApplicationTable.objects.filter(job=job).order_by('-applied_at')[:2]
                result.append({
                    'id': job.id,
                    'job_title': job.title,
                    'skills': job.skills,
                    'posted_at': job.created_at,
                    'applicants': [
                        {
                            'candidate_name': app.candidate.candidate_name,
                            'interview_frames': app.interview_frames,
                            'confidence_score': app.confidence_score,
                            'face_verification_result': app.face_verification_result,
                            'flag_status': app.flag_status,
                            'interview_status': app.interview_status
                        } 
                        for app in applicants
                    ]
                })
            
            return Response(result)
        except Exception as e:
            print(f"Error in JobPostingWithApplicantsView: {str(e)}")
            return Response(
                {"detail": str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
from django.views.generic.list import ListView
from django.http import JsonResponse
from .models import ApplicationTable, JobPostingTable
from django.contrib.auth.mixins import LoginRequiredMixin

class ApplicantsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        try:
            job = JobPostingTable.objects.get(id=job_id, recruiter=request.user.recruiter_profile)
            applicants = ApplicationTable.objects.filter(job=job)
            serializer = ApplicationSerializer(applicants, many=True)
            return Response(serializer.data)
        except JobPostingTable.DoesNotExist:
            return Response({"detail": "Job not found"}, status=status.HTTP_404_NOT_FOUND)
class SortedApplicantsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, job_id):
        try:
            # Get query parameters
            limit = request.query_params.get('limit', '15')
            sort_order = request.query_params.get('sort_order', 'desc')

            # Verify job access
            job = JobPostingTable.objects.get(
                id=job_id, 
                recruiter=request.user.recruiter_profile
            )

            # Get applicants with marks
            applicants = ApplicationTable.objects.filter(job=job).exclude(marks__isnull=True)

            # Convert to list for custom sorting
            applicants_list = list(applicants)

            # Sort based on marks percentage
            def get_percentage(marks_str):
                try:
                    if marks_str and '/' in marks_str:
                        num, den = map(float, marks_str.split('/'))
                        print('num',num)
                        print('den',den)
                        print("Percentages is",(num / den) * 100)
                        return (num / den) * 100
                    return 0
                except:
                    return 0

            applicants_list.sort(
                key=lambda x: get_percentage(x.marks),
                reverse=(sort_order.lower() == 'desc')
            )

            # Apply limit
            if limit.isdigit():
                applicants_list = applicants_list[:int(limit)]

            # Use a simplified serializer for the sorted list
            serializer = SimplifiedApplicantSerializer(applicants_list, many=True)
            return Response(serializer.data)

        except JobPostingTable.DoesNotExist:
            return Response(
                {"detail": "Job not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"detail": str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


from django.conf import settings
import logging
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.utils.html import strip_tags
from django.http import FileResponse  # Add this import

logger = logging.getLogger(__name__)

class ApplicantDetailView(APIView):
    permission_classes = [IsAuthenticated]
    http_method_names = ['get', 'patch']

    def send_status_email(self, applicant, status):
        try:
            subject = "Update on Your Job Application"
            template_name = (
                "emails/selection_email.html" if status == "Selected" 
                else "emails/rejection_email.html"
            )
            
            # Get the recruiter email from the job posting's recruiter
            job_posting = applicant.job
            recruiter = job_posting.recruiter
            # Get the recruiter's email from the User model
            recruiter_email = recruiter.recruiter.email
            
            # Render HTML email template
            html_message = render_to_string(
                template_name,
                {'applicant': applicant, 'recruiter': recruiter,'email': recruiter_email}
            )
            
            # Create plain text version of the email
            plain_message = strip_tags(html_message)
            
            # Create EmailMessage object instead of using send_mail
            from django.core.mail import EmailMessage
            email = EmailMessage(
                subject=subject,
                body=html_message,
                from_email=settings.DEFAULT_FROM_EMAIL,  # This will be your authenticated email
                to=[applicant.email],
            )
            
            # Set reply-to header to the recruiter's email
            email.extra_headers = {'Reply-To': recruiter_email}
            
            # Set the content type to HTML
            email.content_subtype = "html"
            
            # Send the email
            email.send(fail_silently=False)
            
            logger.info(f"Email sent successfully to {applicant.email} with reply-to {recruiter_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

    def patch(self, request, applicant_id):
        try:
            applicant = ApplicationTable.objects.get(
                id=applicant_id,
                job__recruiter=request.user.recruiter_profile
            )
            new_status = request.data.get('application_status')
            reinterview_status = request.data.get('interview_status')
            print("reinterview_status is ",reinterview_status)
            print("new_status of application is ",new_status)
            # Save the status first
             # Update only if the status is provided in the request
            if 'application_status' in request.data:
                applicant.application_status = request.data['application_status']
                applicant.save()


            if 'interview_status' in request.data:
                applicant.interview_status = request.data['interview_status']
                
                # Reset other fields to default/empty values
                applicant.application_status = "Under Review"
                applicant.interview_frames = []  # Empty list for JSONField
                applicant.confidence_score = ""  # Empty string for CharField
                applicant.face_verification_result = []  # Empty list for JSONField
                applicant.flag_status = False  # Reset boolean field
                applicant.request_re_interview = False  # Reset boolean field
                applicant.interview_logs = []  # Empty list for JSONField
                applicant.marks = ""  # Empty string for CharField
                
                applicant.save()


            

            # Then attempt to send email
            if new_status in ["Selected", "Rejected"]:
                email_sent = self.send_status_email(applicant, new_status)
                if not email_sent:
                    # Even if email fails, we still return 200 since status was updated
                    return Response({
                        "detail": "Status updated but email notification failed",
                        "data": DetailedApplicationSerializer(applicant).data
                    }, status=200)

            return Response(DetailedApplicationSerializer(applicant).data)
            
        except ApplicationTable.DoesNotExist:
            return Response({"detail": "Applicant not found"}, status=404)
        except Exception as e:
            logger.error(f"Error in patch: {str(e)}")
            return Response({
                "detail": f"An error occurred: {str(e)}"
            }, status=500)
        
    def get(self, request, applicant_id):
        try:
            applicant = ApplicationTable.objects.get(
                id=applicant_id, 
                job__recruiter=request.user.recruiter_profile
            )
            
            if 'download' in request.query_params:
                return FileResponse(applicant.resume.open(), content_type='application/pdf')
                
            serializer = DetailedApplicationSerializer(applicant)
            return Response(serializer.data)
        except ApplicationTable.DoesNotExist:
            return Response({"detail": "Applicant not found"}, status=status.HTTP_404_NOT_FOUND)
        
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import ApplicationTable, JobPostingTable, CandidateTable
from .serializers import ApplicationTableSerializer
from django.conf import settings
class JobApplicationView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request, *args, **kwargs):
        print("Full request data:", request.data)
        print("Files:", request.FILES)
        
        try:
            job_id = request.data.get("job_id")
            job = JobPostingTable.objects.get(id=job_id)
            candidate = CandidateTable.objects.get(candidate=request.user)
            
            # Check for existing application
            existing_application = ApplicationTable.objects.filter(job=job, candidate=candidate).exists()
            if existing_application:
                return Response(
                    {"error": "You have already applied for this job."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                application = ApplicationTable.objects.create(
                    candidate=candidate,
                    job=job,
                    full_name=request.data.get("full_name"),
                    email=request.data.get("email"),
                    marks=request.data.get("marks", 0),
                    qualification=request.data.get("qualification"),
                    resume=request.FILES.get("resume"),
                )
                print("Application created successfully:", application)
                return Response({"message": "Application submitted successfully."}, status=status.HTTP_201_CREATED)
            except Exception as creation_error:
                import traceback
                print("Error during application creation:", str(creation_error))
                print(traceback.format_exc())
                return Response({"error": str(creation_error)}, status=status.HTTP_400_BAD_REQUEST)

        except JobPostingTable.DoesNotExist:
            return Response({"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND)
        except CandidateTable.DoesNotExist:
            return Response({"error": "Candidate not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            import traceback
            print("General error:", str(e))
            print(traceback.format_exc())
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    


# views.py
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import ApplicationTable, JobPostingTable
from rest_framework.response import Response

class CandidateAppliedJobsView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        try:
            candidate = CandidateTable.objects.get(candidate=user)
            # Get all jobs where the candidate has applications
            applied_jobs = JobPostingTable.objects.filter(
                applicationtable__candidate=candidate
            ).distinct()
            return applied_jobs
        except CandidateTable.DoesNotExist:
            return JobPostingTable.objects.none()

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        data = []
        for job in queryset:
            application = ApplicationTable.objects.filter(
                candidate__candidate=request.user,
                job=job,
                interview_status="not_started"  # Only include applications where interview_status=False

            ).first()
            # If there is no valid application, skip this job
            if not application:
                continue
            
            job_data = {
                'id': job.id,
                'title': job.title,
                'description': job.description,
                'skills': job.skills,
                'experience_level': job.experience_level,
                'location': job.location,
                'status': job.status,
                'created_at': job.created_at,
                'application_date': application.applied_at if application else None,
                'marks': application.marks if application else None,
                'interview_status':application.interview_status,
                'application_status':application.application_status
            }
            data.append(job_data)
            
        return Response(data)

class CandidateInterviewDone(CandidateAppliedJobsView):
    permission_classes=[IsAuthenticated]
    def get_queryset(self):
        queryset = super().get_queryset()
        return queryset
    def list(self, request, *args, **kwargs):
        queryset= self.get_queryset()
        data=[]
        for job in queryset:
            interviews=ApplicationTable.objects.filter(
                candidate__candidate=request.user,
                job=job,
                interview_status='completed'

            ).first()

            if not interviews:
                continue
            interview_data={
                'id':job.id,
                'title': job.title,
                'description' : job.description,
                'skills' : job.description,
                'experience_level' : job.experience_level,
                'location' : job.location,
                'status': job.status,
                'created_at': job.created_at,
                'application_date' : interviews.applied_at if interviews else None,
                'marks': interviews.marks if interviews else None,
                'interview_status' : interviews.interview_status,
                'application_status': interviews.application_status

            }

            data.append(interview_data)
            print('data',data)
        return Response(data)



class CandidateIncompleteInterview(CandidateAppliedJobsView):
    permission_classes=[IsAuthenticated]

    def get_queryset(self):
        queryset =super().get_queryset()
        return queryset
    
    def list(self,request,*args, **kwargs):
        queryset = self.get_queryset()
        data=[]
        for job in queryset:
            incompleted_interviews=ApplicationTable.objects.filter(
                candidate__candidate=request.user,
                job = job,
                interview_status = "started"

            ).first()

            if not incompleted_interviews:
                continue
            
            incomplete_interview_data ={
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "skills": job.skills,
                "experience_level" : job.experience_level,
                "location" : job.location,
                "status" : job.status,
                "created_at" : job.created_at,
                "application_data" : incompleted_interviews.applied_at if incompleted_interviews else None,
                "marks": incompleted_interviews.marks if incompleted_interviews else None,
                'interview_status' : incompleted_interviews.interview_status,
                'application_status': incompleted_interviews.application_status

            }
            data.append(incomplete_interview_data)
        return Response(data)


class RequestReInterview(APIView):
    permission_classes=[IsAuthenticated]
    def post(self,request,interview_id):
        print("request",request,"Job id",interview_id)
        user = self.request.user
        try:
            candidate = CandidateTable.objects.get(candidate=user)
            job = JobPostingTable.objects.get(id=interview_id)
            application = ApplicationTable.objects.get(
                candidate=candidate,
                job=job
            )
            application.request_re_interview=True
            application.save()
            return Response({"message":"Interview request sent successfully."})
        except JobPostingTable.DoesNotExist:
            return Response({"error": "Job not found."}, status=status.HTTP_404_NOT_FOUND)
        except CandidateTable.DoesNotExist:
            return Response({"error": "Candidate not found."}, status=status.HTTP_404_NOT_FOUND)
        except ApplicationTable.DoesNotExist:
            return Response({"error": "Application not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        

from rest_framework.exceptions import NotFound

class CurrentCandidateProfile(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            candidate = CandidateTable.objects.get(candidate=request.user)
            print("image", candidate.profile_image)

            # Get resume URL if exists
            resume_url = candidate.resume.url if candidate.resume and hasattr(candidate.resume, 'url') else None

            # Get profile image URL directly from CloudinaryField
            profile_image_url = candidate.profile_image.url if candidate.profile_image else None

            candidate_data = {
                'id': candidate.id,
                'name': candidate.candidate_name,
                'email': candidate.candidate_email,
                'education': candidate.education,
                'resume': request.build_absolute_uri(resume_url) if resume_url else None,
                'profile_image': profile_image_url,
            }
            return Response(candidate_data)
        except CandidateTable.DoesNotExist:
            raise NotFound("Candidate profile not found.")


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_application_status(request, job_id):
    try:
        application = ApplicationTable.objects.get(
            candidate_id=request.user.candidatetable.id,
            job_id=job_id,
            
        )
        return Response({'status': application.status})
    except ApplicationTable.DoesNotExist:
        return Response({'status': 'Not Applied'}, status=404)
    





# views.py
from django.http import HttpResponse
import zipfile
import os
from io import BytesIO
import json
from django.conf import settings
from .models import ApplicationTable

import requests  # Add this import

def download_frames_as_zip(request, application_id):
    try:
        application = ApplicationTable.objects.get(id=application_id)
        
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            frames = application.face_verification_result
            
            if not frames:
                return HttpResponse("No frames available for this application", status=404)

            unverified_frames = [frame for frame in frames if not frame.get('verified', False)]
            if not unverified_frames:
                return HttpResponse("No unverified frames found for this application", status=404)

            for frame in unverified_frames:
                url = frame.get('url')
                filename = os.path.basename(frame.get('filename', 'image.jpg'))

                if url:
                    response = requests.get(url)
                    if response.status_code == 200:
                        # Write image bytes into the zip file
                        zip_file.writestr(filename, response.content)
                    else:
                        return HttpResponse(f"Failed to download image from {url}", status=500)
        
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer.getvalue(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename=interview_frames_{application_id}.zip'
        return response

    except ApplicationTable.DoesNotExist:
        return HttpResponse("Application not found", status=404)
    except Exception as e:
        return HttpResponse(f"Error: {str(e)}", status=500)



from django.http import HttpResponse
import csv
from io import StringIO
from django.shortcuts import get_object_or_404
from .models import ApplicationTable
import json

def export_interview_logs_csv(request, application_id):
    try:
        # Get the application object
        application = get_object_or_404(ApplicationTable, id=application_id)
        
        # Get the interview logs
        interview_logs = application.interview_logs
        
        # If no logs, return an error
        if not interview_logs or len(interview_logs) == 0:
            return HttpResponse("No interview logs available for this application", status=404)
        
        # Create a CSV file in memory
        csv_buffer = StringIO()
        
        # Determine all possible keys from all entries to create complete headers
        all_keys = set()
        for entry in interview_logs:
            all_keys.update(entry.keys())
        
        # Ensure "question" is the first column in the CSV
        # Create ordered fieldnames with "question" first, then all other keys
        fieldnames = ["question"]
        for key in all_keys:
            if key != "question":
                fieldnames.append(key)
        
        # Create a CSV writer with ordered fieldnames
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write each interview log as a row in the CSV
        for entry in interview_logs:
            # Ensure all keys are present in each row (even if value is empty)
            complete_entry = {key: entry.get(key, '') for key in all_keys}
            writer.writerow(complete_entry)
        
        # Create the HTTP response with CSV content
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=interview_logs_{application_id}.csv'
        
        return response
        
    except Exception as e:
        return HttpResponse(f"Error exporting interview logs: {str(e)}", status=500)


# Add this to handle multiple applications export (for example, all candidates for a job)
def export_all_interview_logs_csv(request, job_id):
    try:
        # Get all applications for the job
        applications = ApplicationTable.objects.filter(job_id=job_id)
        
        # If no applications, return an error
        if not applications.exists():
            return HttpResponse("No applications found for this job", status=404)
        
        # Create a CSV file in memory
        csv_buffer = StringIO()
        
        # Identify columns and their order
        # First two columns will be application_id and candidate_name for identification
        # Then question followed by all other columns
        identifier_columns = ['application_id', 'candidate_name']
        
        # Determine all possible keys from all logs across all applications
        all_keys = set()
        for application in applications:
            for entry in application.interview_logs:
                all_keys.update(entry.keys())
        
        # Create ordered fieldnames starting with identifiers, then question, then others
        fieldnames = identifier_columns + ['question']
        for key in all_keys:
            if key != 'question' and key not in fieldnames:
                fieldnames.append(key)
        
        # Create a CSV writer with ordered fieldnames
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write each interview log as a row in the CSV
        for application in applications:
            candidate_name = f"{application.full_name}" if hasattr(application, 'full_name') else application.candidate_id
            
            for entry in application.interview_logs:
                # Ensure all keys are present in each row (even if value is empty)
                complete_entry = {key: entry.get(key, '') for key in all_keys}
                
                # Add application identifier information
                complete_entry['application_id'] = application.id
                complete_entry['candidate_name'] = candidate_name
                
                writer.writerow(complete_entry)
        
        # Create the HTTP response with CSV content
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=all_interview_logs_job_{job_id}.csv'
        
        return response
        
    except Exception as e:
        return HttpResponse(f"Error exporting interview logs: {str(e)}", status=500)