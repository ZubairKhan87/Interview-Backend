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





from huggingface_hub import InferenceClient

import time
import json

import traceback

# confidence_prediction/views.py
def handle_file(file_path):
    """Helper function to handle file paths for the Hugging Face client"""
    with open(file_path, "rb") as f:
        return f.read()

class ConfidencePredictor:
    def __init__(self):
        # Initialize the Hugging Face client
        self.api_token = os.getenv("HF_API_TOKEN")
        # Model ID (repository name)
        self.model_id = "bairi56/confidence-measure-model"
        
        # We'll try multiple endpoint configurations
        self.endpoints = [
            # Standard HF Inference API endpoint
            f"https://api-inference.huggingface.co/models/{self.model_id}",
            # Gradio Space direct API endpoint - adjust based on actual API path
            "https://bairi56-confidence-measure-model.hf.space/api/predict",
            # Standard Gradio predict endpoint
            "https://bairi56-confidence-measure-model.hf.space/run/predict",
            # Legacy endpoint format
            "https://bairi56-confidence-measure-model.hf.space/predict",
            "https://huggingface.co/spaces/bairi56/confidence-measure-model"
        ]
    
    def process_image_url(self, image_url):
        try:
            print(f"Processing image URL: {image_url}")
            # Download image from URL with a timeout and proper headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(image_url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                print(f"Failed to download image: {response.status_code}")
                return None
            
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            print(f"Sending image to HF model from path: {temp_path}")
            
            # For debugging - we'll hardcode a fixed confidence value if needed
            use_mock_value = os.getenv("USE_MOCK_CONFIDENCE", "").lower() == "true"
            if use_mock_value:
                print("Using mock confidence value for testing")
                return 75.5  # Return a fixed value for testing
            
            try:
                # Try all endpoints until one works
                for endpoint in self.endpoints:
                    try:
                        print(f"Trying endpoint: {endpoint}")
                        
                        # Prepare auth headers if using official API
                        headers = {}
                        if "api-inference.huggingface.co" in endpoint:
                            headers["Authorization"] = f"Bearer {self.api_token}"
                        
                        # Read image for sending
                        with open(temp_path, "rb") as f:
                            image_bytes = f.read()
                        
                        # Try direct API post first
                        api_response = None
                        try:
                            if "run/predict" in endpoint:
                                # For Gradio run/predict endpoint
                                files = {"image": open(temp_path, "rb")}
                                api_response = requests.post(endpoint, files=files, timeout=20)
                            else:
                                # For standard API endpoints
                                api_response = requests.post(
                                    endpoint,
                                    headers=headers,
                                    data=image_bytes,
                                    timeout=20
                                )
                                
                            if api_response and api_response.status_code == 200:
                                print(f"Successful API response from {endpoint}")
                                result = None
                                
                                # Try to parse as JSON, if it fails treat as text
                                try:
                                    result = api_response.json()
                                    print(f"Raw prediction result (JSON): {result}")
                                except json.JSONDecodeError:
                                    result = api_response.text
                                    print(f"Raw prediction result (text): {result}")
                                
                                # Process the result (parsing code moved to separate method)
                                confidence_score = self._parse_result(result)
                                if confidence_score is not None:
                                    return confidence_score
                            else:
                                status_code = api_response.status_code if api_response else "No response"
                                content = api_response.text if api_response else "No content"
                                print(f"API request failed: {status_code} - {content}")
                                
                        except Exception as req_error:
                            print(f"Request to {endpoint} failed: {req_error}")
                    
                    except Exception as endpoint_error:
                        print(f"Error with endpoint {endpoint}: {endpoint_error}")
                
                # If we've tried all endpoints without success, check if model is loaded
                print("All endpoints failed. Attempting to wake up the model...")
                wake_endpoint = "https://bairi56-confidence-measure-model.hf.space/"
                try:
                    wake_response = requests.get(wake_endpoint, timeout=10)
                    print(f"Wake attempt status: {wake_response.status_code}")
                    # Give the model a moment to load
                    import time
                    time.sleep(5)
                    
                    # Try once more with the first endpoint
                    if len(self.endpoints) > 0:
                        print(f"Retrying with endpoint: {self.endpoints[0]}")
                        with open(temp_path, "rb") as f:
                            image_bytes = f.read()
                        
                        headers = {}
                        if "api-inference.huggingface.co" in self.endpoints[0]:
                            headers["Authorization"] = f"Bearer {self.api_token}"
                            
                        retry_response = requests.post(
                            self.endpoints[0],
                            headers=headers,
                            data=image_bytes,
                            timeout=20
                        )
                        
                        if retry_response.status_code == 200:
                            try:
                                result = retry_response.json()
                            except json.JSONDecodeError:
                                result = retry_response.text
                                
                            confidence_score = self._parse_result(result)
                            if confidence_score is not None:
                                return confidence_score
                except Exception as wake_error:
                    print(f"Wake attempt failed: {wake_error}")
                
                # If all else fails, fall back to a random but plausible confidence score
                if os.getenv("USE_FALLBACK_CONFIDENCE", "").lower() == "true":
                    import random
                    fallback_score = random.uniform(60.0, 90.0)
                    print(f"Using fallback confidence score: {fallback_score:.2f}")
                    return round(fallback_score, 2)
                    
                print("All API attempts failed.")
                return None
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_result(self, result):
        """Helper method to parse various result formats and extract confidence score"""
        try:
            # For string results
            if isinstance(result, str):
                # Try to extract percentage with "Confidence: XX%" pattern
                match = re.search(r"Confidence:\s*([\d.]+)%", result)
                if match:
                    confidence_percentage = float(match.group(1))
                    return round(confidence_percentage, 2)
                
                # Try to extract any percentage
                match = re.search(r"([\d.]+)%", result)
                if match:
                    confidence_percentage = float(match.group(1))
                    return round(confidence_percentage, 2)
                
                # Try to extract any number
                match = re.search(r"([\d.]+)", result)
                if match:
                    value = float(match.group(1))
                    # If it's between 0 and 1, assume it's a probability
                    if 0 <= value <= 1:
                        return round(value * 100, 2)
                    return round(value, 2)
                
                # Try parsing as JSON string
                try:
                    json_result = json.loads(result)
                    return self._parse_result(json_result)
                except json.JSONDecodeError:
                    pass
            
            # For Gradio Space API responses - data array format
            elif isinstance(result, dict) and "data" in result:
                data = result["data"]
                if isinstance(data, list) and len(data) > 0:
                    # Try to parse the first item
                    if isinstance(data[0], (int, float)):
                        value = data[0]
                        if 0 <= value <= 1:
                            return round(value * 100, 2)
                        return round(value, 2)
                    
                    # If it's a string, extract numbers
                    elif isinstance(data[0], str):
                        confidence_text = data[0]
                        match = re.search(r"([\d.]+)%", confidence_text)
                        if match:
                            return round(float(match.group(1)), 2)
                        
                        match = re.search(r"([\d.]+)", confidence_text)
                        if match:
                            value = float(match.group(1))
                            if 0 <= value <= 1:
                                return round(value * 100, 2)
                            return round(value, 2)
            
            # For direct list outputs
            elif isinstance(result, list) and len(result) > 0:
                confidence_value = result[0]
                if isinstance(confidence_value, (int, float)):
                    if 0 <= confidence_value <= 1:
                        return round(confidence_value * 100, 2)
                    return round(confidence_value, 2)
                elif isinstance(confidence_value, str):
                    match = re.search(r"([\d.]+)%", confidence_value)
                    if match:
                        return round(float(match.group(1)), 2)
            
            # For dictionary with various confidence keys
            elif isinstance(result, dict):
                # Check various possible keys
                for key in ["confidence", "score", "confidence_score", "prediction", "value", "result"]:
                    if key in result:
                        value = result[key]
                        if isinstance(value, (int, float)):
                            if 0 <= value <= 1:
                                return round(value * 100, 2)
                            return round(value, 2)
                        elif isinstance(value, str):
                            match = re.search(r"([\d.]+)%", value)
                            if match:
                                return round(float(match.group(1)), 2)
                            
                            # Try to extract any number
                            match = re.search(r"([\d.]+)", value)
                            if match:
                                value = float(match.group(1))
                                if 0 <= value <= 1:
                                    return round(value * 100, 2)
                                return round(value, 2)
            
            # If we still haven't found anything useful
            print(f"Could not parse result format: {type(result)} - {result}")
            return None
            
        except Exception as parsing_error:
            print(f"Error parsing result: {parsing_error}")
            return None
                
        # finally:
        #     # Clean up the temporary file
        #     if os.path.exists(temp_path):
        #         os.unlink(temp_path)
                    
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
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

        print(f"Received {len(frames)} frames for analysis")
        
        # Use the HuggingFace-based predictor
        predictor = ConfidencePredictor()
        confidence_scores = []
        errors = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            frame_url = frame.get('url')
            if not frame_url:
                errors.append(f"Frame {i+1} missing URL")
                continue
                
            print(f"Processing frame {i+1}/{len(frames)}: {frame_url}")
            score = predictor.process_image_url(frame_url)
            if score is not None:
                confidence_scores.append(score)
                print(f"✓ Confidence Score for frame {i+1}: {score}")
            else:
                errors.append(f"Failed to process frame {i+1}: {frame_url}")
                print(f"✗ Failed to process frame {i+1}: {frame_url}")
        
        print(f"Successfully processed {len(confidence_scores)}/{len(frames)} frames")
        print(f"Scores: {confidence_scores}")
        
        if not confidence_scores:
            print("No valid predictions obtained")
            return Response({
                'error': 'No valid predictions',
                'details': errors
            }, status=400)
        
        # Calculate the average score
        final_score = sum(confidence_scores) / len(confidence_scores)
        print(f"Average Score: {final_score:.2f} out of 100")
        
        return Response({
            'final_score': round(final_score, 2),
            'individual_scores': confidence_scores,
            'processed_frames': len(confidence_scores),
            'total_frames': len(frames),
            'errors': errors if errors else None
        })
        
    except Exception as e:
        import traceback
        print(f"Error in analyze_confidence: {str(e)}")
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)