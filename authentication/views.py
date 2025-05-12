from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model, authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import IsAuthenticated
from .serializers import CandidateProfileSerializer, RecruiterTableProfileSerializer
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

User = get_user_model()
from django.contrib.auth.models import User
from rest_framework.permissions import AllowAny

from .models import UserProfile
# views.py
# views.py
from google.oauth2 import id_token
from google.auth.transport import requests
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from rest_framework_simplejwt.tokens import RefreshToken
from .models import UserProfile
from django.db import transaction
import random
import string
class GoogleAuthView(APIView):
    permission_classes = [AllowAny]

    def generate_random_password(self, length=12):
        characters = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choice(characters) for _ in range(length))

    def post(self, request):
        try:
            token = request.data.get('token')
            if not token:
                return Response(
                    {'error': 'Token not provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # print("Token from request",token)
            # Verify the Google token
            idinfo = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                settings.GOOGLE_OAUTH2_CLIENT_ID
            )
            # print("Google token is ",idinfo)
            email = idinfo['email']
            name = idinfo.get('name', '')

            # Check if the user exists
            user = User.objects.filter(email=email).first()

            if user:
                # Existing user: Log them in
                refresh = RefreshToken.for_user(user)
                return Response({
                    'access': str(refresh.access_token),
                    'refresh': str(refresh),
                    'username': user.username,
                    'is_recruiter': hasattr(user, 'userprofile') and user.userprofile.is_recruiter,
                }, status=status.HTTP_200_OK)
            else:
                # New user: Create an account
                with transaction.atomic():
                    base_username = email.split('@')[0]
                    username = base_username
                    counter = 1
                    while User.objects.filter(username=username).exists():
                        username = f"{base_username}_{counter}"
                        counter += 1

                    random_password = self.generate_random_password()
                    user = User.objects.create_user(
                        username=username,
                        email=email,
                        password=random_password
                    )

                    # Optional: Add user profile creation logic if needed
                    # UserProfile.objects.create(user=user, is_recruiter=False)

                    # Generate tokens
                    refresh = RefreshToken.for_user(user)
                    return Response({
                        'access': str(refresh.access_token),
                        'refresh': str(refresh),
                        'username': user.username,
                        'is_recruiter': False,
                    }, status=status.HTTP_201_CREATED)

        except ValueError as e:
            print("Token verification error:", str(e))
            return Response(
                {'error': 'Invalid token'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            print("Unexpected error:", str(e))
            return Response(
                {'error': 'Server error occurred'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
class JoinAsRecruiterView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            # Get or create user profile
            user_profile, created = UserProfile.objects.get_or_create(user=request.user)
            
            # Check if user is already a recruiter
            if user_profile.is_recruiter:
                return Response({
                    "message": "You are already a recruiter",
                    "status": "success"
                })
            
            # Request status (pending admin approval)
            return Response({
                "message": "Recruiter access request submitted. Waiting for admin approval.",
                "status": "pending"
            })
        except Exception as e:
            return Response({
                "error": str(e),
                "status": "error"
            }, status=400)

class CheckRecruiterStatusView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        try:
            user_profile = UserProfile.objects.get(user=request.user)
            return Response({
                "is_recruiter": user_profile.is_recruiter
            })
        except UserProfile.DoesNotExist:
            return Response({
                "is_recruiter": False
            })
class RegisterView(APIView):
    # Allow any user (including unauthenticated) to access this view
    permission_classes = [AllowAny]


    def post(self, request):
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')
        confirmPassword = request.data.get('confirmPassword')

        # Check if passwords match
        if password != confirmPassword:
            return Response({"error": "Passwords do not match."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if the username already exists
        if User.objects.filter(username=username).exists():
            return Response({"error": "Username already exists."}, status=status.HTTP_400_BAD_REQUEST)
        
        if User.objects.filter(email=email).exists():
            return Response({"error": "Email already exists."}, status=status.HTTP_400_BAD_REQUEST)

        # Create a new user
        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)


class CandidateLoginView(APIView):
    permission_classes = [AllowAny]
  # Allow unauthenticated access to login

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({
                'error': 'Please provide both username and password'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user = authenticate(username=username, password=password)
        
        if user is None:
            return Response({
                'error': 'Invalid credentials'
            }, status=status.HTTP_401_UNAUTHORIZED)
            
        try:
            profile = UserProfile.objects.get(user=user)
            # Check if user is a recruiter - if so, deny access
            if profile.is_recruiter:
                return Response({
                    'error': 'Please use the recruiter login page'
                }, status=status.HTTP_403_FORBIDDEN)
                
        except UserProfile.DoesNotExist:
            # If no profile exists, create one for candidate
            profile = UserProfile.objects.create(user=user, is_recruiter=False)
        
        # Generate tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'username': user.username,
            'is_recruiter': False
        })

class RecruiterLoginView(APIView):
    permission_classes = [AllowAny]  # Allow unauthenticated access to login

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({
                'error': 'Please provide both username and password'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user = authenticate(username=username, password=password)
        
        if user is None:
            return Response({
                'error': 'Invalid credentials'
            }, status=status.HTTP_401_UNAUTHORIZED)
            
        try:
            profile = UserProfile.objects.get(user=user)
            # Check if user is a recruiter - if not, deny access
            if not profile.is_recruiter:
                return Response({
                    'error': 'Please use the candidate login page'
                }, status=status.HTTP_403_FORBIDDEN)
                
        except UserProfile.DoesNotExist:
            return Response({
                'error': 'Please use the candidate login page'
            }, status=status.HTTP_403_FORBIDDEN)
        
        # Generate tokens
        refresh = RefreshToken.for_user(user)
        
        return Response({
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'username': user.username,
            'is_recruiter': True
        })

from rest_framework_simplejwt.exceptions import TokenError

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")
            if not refresh_token:
                return Response(
                    {"error": "Refresh token is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
                
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response(
                {"message": "Logged out successfully"}, 
                status=status.HTTP_200_OK
            )
        except TokenError:
            # Token is already blacklisted or invalid
            return Response(
                {"message": "Logged out successfully"}, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, 
                status=status.HTTP_400_BAD_REQUEST
            )


class CandidateProfileView(APIView):
    """
    API endpoint for handling candidate profile data.
    GET: Retrieve candidate profile
    PUT/PATCH: Update candidate profile
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    
    def get(self, request):
        """Retrieve the candidate profile data"""
        try:
            # Get the candidate profile for the current authenticated user
            profile = CandidateTable.objects.get(candidate=request.user)
            
            # Serialize the profile data
            serializer = CandidateProfileSerializer(profile)
            
            # Prepare the response with frontend-compatible structure
            response_data = {
                'name': profile.candidate_name,
                'email': profile.candidate_email,
                'headline': profile.about,  # Map about field to headline in frontend
                'location': profile.location,
                'bio': profile.bio,
                'skills': profile.skills,
                'experience': profile.experience,
                'education': profile.education,
                'linkedin': profile.professional_links.get('linkedin', ''),
                'github': profile.professional_links.get('github', ''),
                'portfolio': profile.professional_links.get('portfolio', ''),
                'phone': profile.contact_information.get('phone', ''),
                'joined_date': request.user.date_joined.strftime('%B %d, %Y'),
                # 'profile_image': profile.profile_image
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except CandidateTable.DoesNotExist:
            # Return empty data if the profile doesn't exist yet
            return Response({
                'error': 'Profile not found',
                'message': 'Please complete your profile information'
            }, status=status.HTTP_404_NOT_FOUND)
    
    def put(self, request):
        """Update the candidate profile data"""
        try:
            # Get the existing profile or create a new one
            profile, created = CandidateTable.objects.get_or_create(
                candidate=request.user,
                defaults={
                    'candidate_name': request.user.get_full_name() or request.user.username,
                    'candidate_email': request.user.email,
                    'skills': [],
                    'professional_links': {},
                    'contact_information': {}
                }
            )
            
            # Process the incoming data
            data = request.data
            print("data", data)
            
            # Process the main fields
            if 'name' in data:
                profile.candidate_name = data['name']
            if 'email' in data:
                profile.candidate_email = data['email']
            if 'headline' in data:
                profile.about = data['headline']
            if 'location' in data:
                profile.location = data['location']
            if 'bio' in data:
                profile.bio = data['bio']
            if 'experience' in data:
                profile.experience = data['experience']
            if 'education' in data:
                profile.education = data['education']
            
            # Handle JSONField for skills properly
            if 'skills' in data:
                # Ensure it's a list type when saving
                if isinstance(data['skills'], list):
                    profile.skills = data['skills']
                else:
                    # Handle if it's sent as a string or other format
                    try:
                        # If it might be a JSON string, try to parse it
                        import json
                        profile.skills = json.loads(data['skills'])
                    except (TypeError, json.JSONDecodeError):
                        # If parsing fails, store as a list with one item
                        profile.skills = [data['skills']]
            
            if 'profile_image' in data:
                profile.profile_image = data['profile_image']
            
            # Reset professional_links if it's not a dict
            if not isinstance(profile.professional_links, dict):
                profile.professional_links = {}
                
            # Process professional links
            professional_links = dict(profile.professional_links)  # Create a new dict to be safe
            if 'linkedin' in data:
                professional_links['linkedin'] = data['linkedin']
            if 'github' in data:
                professional_links['github'] = data['github']
            if 'portfolio' in data:
                professional_links['portfolio'] = data['portfolio']
            profile.professional_links = professional_links
            
            # Reset contact_information if it's not a dict
            if not isinstance(profile.contact_information, dict):
                profile.contact_information = {}
                
            # Process contact information
            contact_info = dict(profile.contact_information)  # Create a new dict to be safe
            if 'phone' in data:
                contact_info['phone'] = data['phone']
            profile.contact_information = contact_info
            
            # Process resume file if provided
            if 'resume' in request.FILES:
                profile.resume = request.FILES['resume']
            
            # Save the updated profile
            profile.save()
            
            return Response({
                'message': 'Profile updated successfully',
                'data': CandidateProfileSerializer(profile).data
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            print(f"Error updating profile: {str(e)}")
            print(traceback.format_exc())  # Print full traceback for debugging
            return Response({
                'error': 'Failed to update profile',
                'details': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

class RecruiterProfileView(APIView):
    """
    API endpoint for handling recruiter profile data.
    GET: Retrieve recruiter profile
    PUT/PATCH: Update recruiter profile
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    
    def get(self, request):
        """Retrieve the recruiter profile data"""
        try:
            # Get the recruiter profile for the current authenticated user
            profile = RecruiterTable.objects.get(recruiter=request.user)
            
            # Prepare the response with frontend-compatible structure
            response_data = {
                'name': profile.recruiter_name,
                'email': request.user.email,
                'headline': profile.about,  # Map about field to headline in frontend
                'location': profile.location,
                'bio': profile.bio,
                'phone': profile.contact_information.get('phone', ''),
                'joined_date': request.user.date_joined.strftime('%B %d, %Y'),
                
                # Company information
                'company': profile.recruiter_organization,
                'company_website': profile.company_information.get('company_website', ''),
                'industry': profile.company_information.get('industry', ''),
                'company_size': profile.company_information.get('company_size', ''),
                
                # Professional links
                'linkedin': profile.company_information.get('linkedin', ''),
                'company_linkedin': profile.company_information.get('company_linkedin', ''),
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except RecruiterTable.DoesNotExist:
            # Return empty data if the profile doesn't exist yet
            return Response({
                'error': 'Profile not found',
                'message': 'Please complete your profile information'
            }, status=status.HTTP_404_NOT_FOUND)
    
    def put(self, request):
        """Update the recruiter profile data"""
        try:
            # Get the existing profile or create a new one
            profile, created = RecruiterTable.objects.get_or_create(
                recruiter=request.user,
                defaults={
                    'recruiter_name': request.user.get_full_name() or request.user.username,
                    'recruiter_organization': '',
                    'location': '',
                    'bio': '',
                    'about': '',
                    'company_information': {},
                    'contact_information': {}
                }
            )
            
            # Process the incoming data
            data = request.data
            print("Received profile update data:", data)
            
            # Process the main fields
            if 'name' in data:
                profile.recruiter_name = data['name']
            if 'headline' in data:
                profile.about = data['headline']
            if 'location' in data:
                profile.location = data['location']
            if 'bio' in data:
                profile.bio = data['bio']
            
            # Process profile image if provided
            if 'profile_image' in request.FILES:
                # Assuming you have a function to handle file uploads and return URLs
                profile.profile_image = handle_uploaded_image(request.FILES['profile_image'])
            
            # Process company information
            company_info = dict(profile.company_information) if isinstance(profile.company_information, dict) else {}
            
            if 'company' in data:
                profile.recruiter_organization = data['company']
            if 'company_website' in data:
                company_info['company_website'] = data['company_website']
            if 'industry' in data:
                company_info['industry'] = data['industry']
            if 'company_size' in data:
                company_info['company_size'] = data['company_size']
            
            # Process professional links
            if 'linkedin' in data:
                company_info['linkedin'] = data['linkedin']
            if 'company_linkedin' in data:
                company_info['company_linkedin'] = data['company_linkedin']
            
            # Update the company_information field
            profile.company_information = company_info
            
            # Process contact information
            contact_info = dict(profile.contact_information) if isinstance(profile.contact_information, dict) else {}
            if 'phone' in data:
                contact_info['phone'] = data['phone']
            profile.contact_information = contact_info
            
            # Save the updated profile
            profile.save()
            
            return Response({
                'message': 'Profile updated successfully',
                'data': {
                    'name': profile.recruiter_name,
                    'email': request.user.email,
                    'headline': profile.about,
                    'location': profile.location,
                    'bio': profile.bio,
                    'company': profile.recruiter_organization,
                    # Include other fields as needed
                }
            }, status=status.HTTP_200_OK)
            
        except Exception as e:
            import traceback
            print(f"Error updating recruiter profile: {str(e)}")
            print(traceback.format_exc())  # Print full traceback for debugging
            return Response({
                'error': 'Failed to update profile',
                'details': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
        

class CandidateResumeUploadView(APIView):
    """API endpoint for updating candidate resume only"""
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]
    
    def post(self, request):
        try:
            profile, created = CandidateTable.objects.get_or_create(
                candidate=request.user,
                defaults={
                    'candidate_name': request.user.get_full_name() or request.user.username,
                    'candidate_email': request.user.email,
                    'skills': [],
                    'professional_links': {},
                    'contact_information': {}
                }
            )
            
            if 'resume' in request.FILES:
                profile.resume = request.FILES['resume']
                profile.save()
                
                return Response({
                    'message': 'Resume updated successfully',
                    'resume_url': profile.resume.url
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'error': 'No resume file provided'
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Exception as e:
            return Response({
                'error': 'Failed to update resume',
                'details': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import json



# views.py
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import CandidateTable

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_candidate_profile_image(request):
    print("Logged-in user:", request.user)

    try:
        candidate = CandidateTable.objects.get(candidate=request.user)
        image_url = candidate.profile_image.url if candidate.profile_image else None
        print("image url candidate",image_url)
        return Response({
            'profile_image': image_url,
            'candidate_name': candidate.candidate_name
        })

    except CandidateTable.DoesNotExist:
        return Response({'error': 'Profile not found'}, status=status.HTTP_404_NOT_FOUND)

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import parser_classes

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_candidate_profile_image(request):
    try:
        candidate = CandidateTable.objects.get(candidate=request.user)
        
        if candidate.profile_image:
            return Response(
                {'error': 'Profile image already exists and cannot be modified'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        image_file = request.FILES.get('profile_image')
        if not image_file:
            return Response({'error': 'Profile image is required'}, status=400)

        # Directly assign image file to CloudinaryField
        candidate.profile_image = image_file
        candidate.save()

        return Response({
            'message': 'Profile image uploaded successfully',
            'profile_image': candidate.profile_image.url
        })

    except CandidateTable.DoesNotExist:
        return Response({'error': 'Profile not found'}, status=status.HTTP_404_NOT_FOUND)
    

from .models import RecruiterTable
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_recruiter_profile_image(request):
    print("Logged-in user:", request.user)

    try:
        recruiter = RecruiterTable.objects.get(recruiter=request.user)
        image_url = recruiter.profile_image.url if recruiter.profile_image else None

        return Response({
            'profile_image': image_url,
            'recruiter_name': recruiter.recruiter_name
        })

    except RecruiterTable.DoesNotExist:
        return Response({'error': 'Profile not found'}, status=status.HTTP_404_NOT_FOUND)

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import parser_classes

@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_recruiter_profile_image(request):
    try:
        recruiter = RecruiterTable.objects.get(recruiter=request.user)
        
        image_file = request.FILES.get('profile_image')
        if not image_file:
            return Response({'error': 'Profile image is required'}, status=400)

        # Direct assignment to CloudinaryField
        recruiter.profile_image = image_file
        recruiter.save()

        return Response({
            'message': 'Profile image uploaded successfully',
            'profile_image': recruiter.profile_image.url
        })

    except RecruiterTable.DoesNotExist:
        return Response({'error': 'Profile not found'}, status=status.HTTP_404_NOT_FOUND)
from django.core.mail import send_mail
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import logging

logger = logging.getLogger(__name__)

@api_view(['POST'])
@permission_classes([AllowAny])

def recruiter_registration_request(request):
    try:
        # Extract data from request
        data = request.data
        username = data.get('username')
        email = data.get('email')
        organization = data.get('organization')
        # password = data.get('password')
        
        # Validate required fields
        if not all([username, email, organization]):
            return Response(
                {"error": "All fields are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Compose email content
        subject = f"New Recruiter Registration Request - {organization}"
        message = f"""
New Recruiter Registration Request

Details:
- Username: {username}
- Email: {email}
- Organization: {organization}

This recruiter has requested to join the AI-Powered Interviewing System.
Please review their application and create their account if approved.

Best regards,
AI Interviewing System
        """
        
        # Send email to all admin emails
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=settings.ADMIN_EMAILS,  # Now using ADMIN_EMAILS list instead of single ADMIN_EMAIL
            fail_silently=False,
        )
        
        logger.info(f"Recruiter registration request received for {email}")
        
        return Response(
            {"message": "Registration request submitted successfully"},
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Error processing recruiter registration: {str(e)}")
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )




# In one of your apps' views.py or create a new file health_check.py
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "healthy"})