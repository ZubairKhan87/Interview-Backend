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
            print("img_array",img_array)
            return img_array

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

    def verify_faces(self, ref_img_path, target_img_path): 
        try:
            # Initialize Hugging Face token
            hf_token = os.getenv("HF_API_TOKEN")
            if not hf_token:
                logger.error("HF_API_TOKEN not found in environment")
                return {"error": "API configuration error. Please contact support."}

            # Preprocess images
            ref_img = self.preprocess_image(ref_img_path)
            target_img = self.preprocess_image(target_img_path)
            print("ref_img",ref_img)
            print("target_img",target_img)
            # Save images temporarily
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)

            ref_temp = os.path.join(temp_dir, f"ref_{uuid.uuid4().hex}.jpg")
            target_temp = os.path.join(temp_dir, f"target_{uuid.uuid4().hex}.jpg")

            Image.fromarray(ref_img).save(ref_temp, format='JPEG', quality=95)
            Image.fromarray(target_img).save(target_temp, format='JPEG', quality=95)

            # Initialize Gradio client
            client = Client("bairi56/face-verification", hf_token=hf_token)
            print("client",client)
            # Send API request
            result = client.predict(
                img1=handle_file(ref_temp),
                img2=handle_file(target_temp),
                api_name="/predict"
            )
            print("result",result)
            logger.error(f"HF API Result: {result}")

            if isinstance(result, str):

                match = re.search(r"Similarity Score: (\d+\.\d+)", result)
                similarity = float(match.group(1)) if match else None
                print("similarity",similarity)
                return {
                    "match": "Match: Yes" in result or "✅" in result,
                    "confidence": round(similarity * 100, 2) if similarity else None,
                    "raw_response": result
                }
            else:
                return {"error": "Unexpected response format from verification model."}

        except Exception as e:
            logger.error(f"Face verification error: {str(e)}")
            return {"error": f"Verification failed: {str(e)}"}

        finally:
            # Clean up
            for file in [ref_temp, target_temp]:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except Exception:
                    pass


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
                print("verification_result",verification_result)
                logger.info(f"Verification Result: {verification_result}")
                if "error" in verification_result:
                    return Response("error in verification",verification_result, status=status.HTTP_400_BAD_REQUEST)

                return Response(verification_result)

            finally:
                # Clean up reference image
                try:
                    if os.path.exists(ref_image_path):
                        os.remove(ref_image_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Verification process failed......: {str(e)}")
            return Response(
                {"error": f"Verification process failed !!!!: {str(e)}"},
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






import time

import traceback
# confidence_prediction/views.py
# Fixed version of your confidence prediction code

import os
import re
import tempfile
import traceback
import requests
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

class ConfidencePredictor:
    def __init__(self):
        self.api_token = os.getenv("HF_API_TOKEN")
        print("api_token",self.api_token)
        if not self.api_token:
            print("HF_API_TOKEN not set")
            self.client = None
            return
        self.client = Client("bairi56/confidence-measure-model", hf_token=self.api_token)  # Changed to token parameter
        print("client........",self.client)
    def process_image_url(self, image_url):
        try:
            import re
            from gradio_client import handle_file
            
            # Use handle_file directly with the URL as shown in your example
            # This is the key change - handle_file can process URLs directly
            result = self.client.predict(
                image=handle_file(image_url),
                api_name="/predict"
            )
            
            print(f"Raw result: {result}")
            
            # Parse the result based on its format
            if isinstance(result, str):
                match = re.search(r"Confidence:\s*([\d.]+)%", result)
                return round(float(match.group(1)), 2) if match else None
            elif isinstance(result, (list, tuple)) and result:
                val = result[0]
                return round(val * 100, 2) if 0 <= val <= 1 else round(val, 2)
            elif isinstance(result, dict) and 'confidence' in result:
                return round(float(result['confidence']) * 100, 2)
            
            print(f"Unexpected result format: {type(result)}, content: {result}")
            return None
            
        except Exception as e:
            import traceback
            print(f"Error processing image: {e}")
            traceback.print_exc()
            return None

def clean_url(url):
    """
    Clean the URL by removing surrounding quotes or trailing characters.
    """
    cleaned = url.strip().strip('\'"')
    return cleaned

def is_url_valid(url):
    """
    Check if the URL is reachable (returns HTTP 200).
    """
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"URL check failed: {url}, Error: {str(e)}")
        return False

# The key fix is below - don't wrap the view with @api_view multiple times
import json
@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
def analyze_confidence(request):
    try:
        frames = []
        if isinstance(request.data, dict):
            frames = request.data.get('frames', [])
        elif isinstance(request.data, str):
            try:
                data = json.loads(request.data)
                frames = data.get('frames', [])
            except json.JSONDecodeError:
                return Response({'error': 'Invalid JSON data'}, status=400)

        if not frames:
            return Response({'error': 'No frames provided'}, status=400)

        for frame in frames:
            if isinstance(frame, dict) and 'url' in frame:
                frame['url'] = clean_url(frame['url'])

        predictor = ConfidencePredictor()
        if not predictor.client:
            return Response({'error': 'HuggingFace model not initialized'}, status=500)

        confidence_scores = []

        for frame in frames:
            url = frame.get("url")
            print(f"Processing: {url}")
            if is_url_valid(url):
                score = predictor.process_image_url(url)
                if score is not None:
                    confidence_scores.append(score)
            else:
                print(f"Invalid URL skipped: {url}")

        if not confidence_scores:
            return Response({'error': 'No valid predictions'}, status=400)

        avg_score = round(sum(confidence_scores) / len(confidence_scores), 2)
        return Response({'final_score': avg_score})

    except Exception as e:
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)
