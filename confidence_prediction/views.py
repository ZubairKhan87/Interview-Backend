# views.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 👈 Disable GPU checks (MUST be before tensorflow/deepface imports)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
import numpy as np
import os
from django.conf import settings
import logging
import os
import requests
from io import BytesIO
import threading
from gradio_client import Client, handle_file
import tempfile
import re
import cloudinary
import cloudinary.uploader
import requests
import re
from PIL import Image, ImageOps
import numpy as np
logger = logging.getLogger(__name__)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

logger = logging.getLogger(__name__)
import requests
import uuid
logger = logging.getLogger(__name__)
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
class FaceVerificationView(APIView):
    permission_classes = [IsAuthenticated]
    TARGET_SIZE = (640, 480)  # Standard size for processing

    def __init__(self):
        super().__init__()

    def get_image_path(self, image_url):
        """Download and return local path of the profile image (Cloudinary or media)."""
        if not image_url or image_url == 'null':
            raise ValueError("No profile image URL provided")

        try:
            # Check if it's a Cloudinary/external URL
            if image_url.startswith('http'):
                temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
                os.makedirs(temp_dir, exist_ok=True)

                filename = f"cloud_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(temp_dir, filename)
                
                # Download the image
                response = requests.get(image_url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to download image from {image_url}")
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path

            # Handle local media path
            if '/media/' in image_url:
                media_index = image_url.find('/media/')
                path_after_media = image_url[media_index + 7:]
                absolute_path = os.path.join(settings.MEDIA_ROOT, path_after_media)
                if not os.path.exists(absolute_path):
                    raise ValueError(f"Profile image not found at: {absolute_path}")
                return absolute_path

            raise ValueError(f"Unrecognized image URL format: {image_url}")

        except Exception as e:
            logger.error("Error processing image path: %s", str(e))
            raise ValueError(f"Error processing image path: {str(e)}")

    def preprocess_image(self, image_path):
        """Preprocess image for consistent size and format using Pillow."""
        try:
            # Open the image and convert to RGB
            img = Image.open(image_path).convert('RGB')

            # Resize while maintaining aspect ratio and padding
            target_w, target_h = self.TARGET_SIZE
            img_resized = ImageOps.pad(img, (target_w, target_h), color=(0, 0, 0), centering=(0.5, 0.5))

            # Convert to NumPy array
            img_array = np.array(img_resized)

            return img_array

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def verify_faces(self, ref_img_path, target_img_path): 
        """Verifies whether the faces in two images belong to the same person using deployed HF API."""
        try:
            # Preprocess images
            ref_img = self.preprocess_image(ref_img_path)
            target_img = self.preprocess_image(target_img_path)

            # Save preprocessed images temporarily
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            temp_ref_path = os.path.join(temp_dir, f'temp_ref_{uuid.uuid4().hex}.jpg')
            temp_target_path = os.path.join(temp_dir, f'temp_target_{uuid.uuid4().hex}.jpg')

            # Save temp files
            Image.fromarray(ref_img).save(temp_ref_path, format="JPEG", quality=95)
            Image.fromarray(target_img).save(temp_target_path, format="JPEG", quality=95)
            
            try:
                # APPROACH: Use direct HTTP requests instead of gradio_client
                # This is a fallback method since gradio_client is returning HTML instead of JSON
                
                hf_token = os.getenv("HF_API_TOKEN")
                if not hf_token:
                    logger.error("HF_API_TOKEN not found in environment variables")
                    # Use hardcoded token as fallback for testing only (remove in production)
                    hf_token = "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"  # This is from your test code
                    logger.warning("Using fallback token for testing. Remove in production.")
                
                logger.info("Using direct HTTP requests instead of gradio_client")
                
                # Direct API access using requests
                import requests
                
                # Define API URL - using the endpoint that worked in your direct check
                api_url = "https://bairi56-face-verification.hf.space/run/predict"
                
                # Prepare files for multipart upload
                files = {
                    'img1': ('ref_image.jpg', open(temp_ref_path, 'rb'), 'image/jpeg'),
                    'img2': ('target_image.jpg', open(temp_target_path, 'rb'), 'image/jpeg')
                }
                
                # Prepare headers with authentication
                headers = {
                    'Authorization': f'Bearer {hf_token}'
                }
                
                # Make the API request
                logger.info(f"Sending direct HTTP request to {api_url}")
                response = requests.post(api_url, files=files, headers=headers)
                
                # Check response status
                logger.info(f"API response status: {response.status_code}")
                logger.info(f"API response headers: {response.headers}")
                
                # Try to parse the response
                if response.status_code == 200:
                    try:
                        # Try JSON first
                        result_data = response.json()
                        logger.info(f"Received JSON response: {result_data}")
                        
                        # Extract the result (structure depends on API)
                        result = result_data.get('data', '')
                        if isinstance(result, list) and len(result) > 0:
                            result = result[0]  # Common format for HF API responses
                            
                    except Exception as json_err:
                        # Fall back to text response
                        logger.warning(f"JSON parsing failed: {str(json_err)}")
                        result = response.text
                        logger.info(f"Using text response: {result[:200]}...")  # Log first 200 chars
                else:
                    return {"error": f"API request failed with status code: {response.status_code}"}
                
                # Process the result
                if isinstance(result, str):
                    if "Match: Yes" in result or "Match ✅" in result:
                        # Extract similarity score
                        similarity = None
                        try:
                            # Look for "Similarity Score: X.XXXX" pattern
                            match = re.search(r"Similarity Score: (\d+\.\d+)", result)
                            if match:
                                similarity = float(match.group(1))
                        except Exception as e:
                            logger.warning(f"Failed to extract similarity score: {str(e)}")
                            
                        return {
                            "match": True,
                            "confidence": round(similarity * 100, 2) if similarity else None,
                            "raw_response": result  # Include raw response for debugging
                        }
                        
                    elif "too dark or unclear" in result:
                        return {"error": "Face in Target Image is too dark or unclear. Ensure good lighting and look directly at the camera."}
                    elif "No face detected" in result:
                        return {"error": "No face detected. Please ensure your face is clearly visible."}
                    elif "Multiple faces detected" in result:
                        return {"error": "Multiple faces detected. Please ensure only your face is in the frame."}
                    elif "Face not clear enough" in result:
                        return {"error": "Face not clear enough. Please ensure good lighting and look directly at the camera."}
                    else:
                        # No match but valid response
                        return {
                            "match": False,
                            "raw_response": result  # Include raw response for debugging
                        }
                else:
                    # Attempt to extract information from other response formats
                    logger.info(f"Processing non-string result: {type(result)}")
                    logger.info(f"Result content: {result}")
                    
                    # Try to determine if there's a match based on the response structure
                    # This will depend on the actual response format of your API
                    match_found = False
                    confidence = None
                    
                    # Add logic here to extract match information from the response
                    # (This is placeholder logic and should be adjusted based on actual response)
                    if isinstance(result, dict):
                        if result.get('match') is not None:
                            match_found = result.get('match')
                        if result.get('confidence') is not None:
                            confidence = result.get('confidence')
                        if result.get('similarity') is not None:
                            confidence = float(result.get('similarity')) * 100
                    
                    return {
                        "match": match_found,
                        "confidence": confidence,
                        "raw_response": str(result)  # Convert to string for safe logging
                    }
                    
            except Exception as e:
                logger.error(f"Verification error: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"error": f"Verification failed: {str(e)}"}

            finally:
                # Clean up
                for path in [temp_ref_path, temp_target_path]:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Verification error: {str(e)}")
            return {"error": f"Verification failed: {str(e)}"}
    def post(self, request):
        try:
            # Get and validate reference image
            ref_image = request.FILES.get('ref_image')
            if not ref_image:
                return Response(
                    {"error": "No reference image provided"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Get and validate target image path
            target_image = request.data.get('target_image')
            if not target_image or target_image == 'null':
                return Response(
                    {"error": "No profile image found. Please upload your profile image first."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                target_image_path = self.get_image_path(target_image)
            except ValueError as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            # Save reference image with unique name
            ref_image_path = os.path.join(temp_dir, f'ref_{request.user.id}_{uuid.uuid4().hex}_{ref_image.name}')
            with open(ref_image_path, 'wb+') as destination:
                for chunk in ref_image.chunks():
                    destination.write(chunk)

            try:
                # Verify faces
                verification_result = self.verify_faces(ref_image_path, target_image_path)
                if "error" in verification_result:
                    return Response(verification_result, status=status.HTTP_400_BAD_REQUEST)

                return Response(verification_result)

            finally:
                # Clean up reference image
                try:
                    if os.path.exists(ref_image_path):
                        os.remove(ref_image_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Verification process failed: {str(e)}")
            return Response(
                {"error": f"Verification process failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


from rest_framework.permissions import IsAuthenticated,AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
logger = logging.getLogger(__name__)
@method_decorator(csrf_exempt, name='dispatch')
class FaceVerificationCheat(APIView):
    permission_classes = [AllowAny]

    def __init__(self):
        super().__init__()

    def get_image_path(self, image_url):
        """Download and return local path of the profile image (Cloudinary or media)."""
        if not image_url or image_url == 'null':
            raise ValueError("No profile image URL provided")

        try:
            # Check if it's a Cloudinary/external URL
            if image_url.startswith('http'):
                print("inside image_url condition",image_url)
                temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
                os.makedirs(temp_dir, exist_ok=True)

                filename = f"cloud_{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(temp_dir, filename)
                print("file_path",file_path)
                # Download the image
                response = requests.get(image_url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to download image from {image_url}")
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print("file_path at return",file_path)
                return file_path

            # Handle local media path
            if '/media/' in image_url:
                media_index = image_url.find('/media/')
                path_after_media = image_url[media_index + 7:]
                absolute_path = os.path.join(settings.MEDIA_ROOT, path_after_media)
                if not os.path.exists(absolute_path):
                    raise ValueError(f"Profile image not found at: {absolute_path}")
                return absolute_path

            raise ValueError(f"Unrecognized image URL format: {image_url}")

        except Exception as e:
            logger.error("Error processing image path: %s", str(e))
            raise ValueError(f"Error processing image path: {str(e)}")


    def verify_single_frame(self, frame_path, target_image_path):
        """Verify a single frame against the target image, returning only True/False."""
        try:
            print("Single frame condition is running")

            # client = Client("bairi56/face-verification")
            client = Client(
                "bairi56/face-verification",
                hf_token=os.getenv("HF_API_TOKEN"),  # Add this line


            )

            result = client.predict(
                frame_path,
                target_image_path,
                api_name="/predict"
            )

            # If it's a match, return True
            if isinstance(result, str) and "Match ✅" in result:
                return True
            return False

        except Exception:
            return False


    def post(self, request):
        """
        Handle verification of multiple frames against a profile image.
        Expected input:
        {
            "profile_image": "/media/profile_image.jpg",
            "frames": [
                {"url": "/media/frame1.jpg"},
                {"url": "/media/frame2.jpg"},
                ...
            ]
        }
        """
        try:
            print("inside face verification post method in try condition")
            # Get profile image path
            profile_image = request.data.get('profile_image')
            if not profile_image:
                return Response({"error": "No profile image provided"}, status=status.HTTP_400_BAD_REQUEST)
            print("profile image in face verification",profile_image)
            profile_image_path = self.get_image_path(profile_image)
            print("profile_image_path in face verification",profile_image_path)
            if not profile_image_path:
                return Response({"error": "Invalid profile image path"}, status=status.HTTP_400_BAD_REQUEST)

            # Get frames
            frames = request.data.get('frames', [])
            print("frames in face verification",frames)
            if not frames:
                return Response({"error": "No frames provided"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Process each frame
            verification_results = []
            for frame in frames:
                frame_url = frame.get('url')
                frame_path = self.get_image_path(frame_url)
                
                if frame_path:
                    print("frame_path in face verification ,if condition fot true",frame_path)
                    is_verified = self.verify_single_frame(frame_path, profile_image_path)
                else:
                    is_verified = False
                
                verification_results.append({
                    "frame_url": frame_url,
                    "verified": is_verified
                })
                # logger.info(f" Face Verification Results: {verification_results}")

                # print("verification_results",verification_results)

            print("verification_results",verification_results)

            return Response({
                "verification_results": verification_results,
                "summary": {
                    "total_frames": len(frames),
                    "verified_frames": sum(1 for result in verification_results if result["verified"]),
                    "verification_rate": round(sum(1 for result in verification_results if result["verified"]) / len(frames) * 100, 2)
                }
            })

        except Exception as e:
            logger.error(f"Verification process failed: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)








# confidence_prediction/views.py
class ConfidencePredictor:
    def __init__(self):
        # Initialize the Hugging Face client
        self.api_token =os.getenv("HF_API_TOKEN")
        self.client = Client(
            "bairi56/confidence-measure-model",
            hf_token=self.api_token
        )

    def process_image_url(self, image_url):
        try:
            # Download image from URL
            response = requests.get(image_url)
            if response.status_code != 200:
                print(f"Failed to download image: {response.status_code}")
                return None

            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name

            try:
                # Make prediction using the Hugging Face model
                result = self.client.predict(
                    image=temp_path,      # Just pass the path directly
                    api_name="/predict"
                )

                
                
                # Process the result based on  model's output format
                if isinstance(result, str):
                    # Try to extract the confidence value from the string using regex
                    match = re.search(r"Confidence:\s*([\d.]+)%", result)
                    if match:
                        confidence_percentage = float(match.group(1))
                        return round(confidence_percentage, 2)
                    else:
                        print(f"Could not extract confidence value from: {result}")
                        return None

                elif isinstance(result, (list, tuple)) and len(result) > 0:
                    confidence_value = result[0]
                    if isinstance(confidence_value, (int, float)):
                        if 0 <= confidence_value <= 1:
                            return round(confidence_value * 100, 2)
                        return round(confidence_value, 2)
                else:
                    print(f"Unexpected result format: {result}")
                    return None
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
@api_view(['POST'])
@permission_classes([AllowAny])  # Require authentication
def analyze_confidence(request):
    try:
        frames = request.data.get('frames', [])
        if not frames:
            return Response({'error': 'No frames provided'}, status=400)

        # Use the new HuggingFace-based predictor
        predictor = ConfidencePredictor()
        confidence_scores = []
        
        # Process each frame
        for frame in frames:
            frame_url = frame.get('url')
            score = predictor.process_image_url(frame_url)
            if score is not None:
                confidence_scores.append(score)
                print("Confidence Scores of frame :", score)
        
        if not confidence_scores:
            return Response({'error': 'No valid predictions'}, status=400)
        
        # Calculate the average score
        final_score = sum(confidence_scores) / len(confidence_scores)

        print(f"Average Score: {final_score:.2f} out of 100")
        return Response({
            'final_score': final_score
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=500)