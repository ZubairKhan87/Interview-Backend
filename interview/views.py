import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 👈 Disable G
from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import re
import csv
import os
from datetime import datetime
from django.conf import settings
from corsheaders.defaults import default_headers
from django.shortcuts import render, get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from huggingface_hub import InferenceClient
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from job_posting.models import JobPostingTable, ApplicationTable
from authentication.models import CandidateTable
from groq import Groq

from pathlib import Path
from backend.settings import BASE_URL
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
# print(huggingface_hub._version_)
# Initialize the inference client
# client = Groq(api_key=os.getenv('GROQ_API_KEY'))
client=Groq(api_key="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")

# job = "Python programmer"
# Skills="Object oriented programming, functions, and Data structures"
# candidate_name="Faizaaaan"
# experience="Fresher"
# from sapling import SaplingClient

# def AI_Detetection(text_to_analyze):
#     api_key =os.getenv('SAMPLE_API_KEY')
#     client = SaplingClient(api_key=api_key)
#     # text_to_analyze = ""
#     detection_results = client.aidetect(text_to_analyze, sent_scores=True)

#     # Evaluate the AI score
#     score = detection_results.get('score', 0)
#     # print(score)
#     # if score > 0.5:
#     #     print("The text is likely AI-generated.")
#     # elif score < 0.3:
#     #     print("The text is likely human-written.")
#     # else:
#     #     print("Uncertain. Manual review recommended.")

#     print(f"AI Probability Score: {score}")
#     return score

# Helper function to get interview details from database
def get_interview_details(job_id, candidate_id):
    """
    Fetches job and candidate details from database for the interview.
    """
    job = get_object_or_404(JobPostingTable, id=job_id)
    candidate = get_object_or_404(CandidateTable, id=candidate_id)
    
    # Get the application
    application = get_object_or_404(ApplicationTable, job=job, candidate=candidate)
    
    # Convert skills list to string format for LLM
    base_skills = "Object oriented programming, Data structures and Databases"
    job_specific_skills = ", ".join(job.skills)  # Assuming skills is stored as JSON array
    combined_skills = f"{base_skills}, {job_specific_skills}"
    
    return {
        'job': job.title,
        'skills': combined_skills,
        'experience': job.experience_level,
        'candidate_name': application.full_name,
        'profile_image':candidate.profile_image,
        'interview_frames': application.interview_frames
    }

def get_intent(response):
    """
    Sends the candidate's response to the LM for intent classification.
    """
    intent_prompt = f"""
    The following is a response from a candidate during a job interview. Just return the intent of their response in single word into one of these categories:
    --> Telling_Personal_Info
    --> Answering_to_a_technical_question_in_an_interview
    --> Quit_interview
    --> Asking_for_clarification_or_hint
    --> Others
    Candidate's response: "{response}"
    Intent: ?
    Return exact intent, not its number"""
    try:
        # Generate intent classification
        intent_response =client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "user",
                    "content": intent_prompt
                },
            ],
            temperature=1,
            # max_completion_tokens=1024,
            top_p=1,
            # stream=True,
            stop=None,
        )
        # Extract intent from the LM's response
        # intent_message = intent_response.choices[0].message['content'].strip()
        intent_message = intent_response.choices[0].message.content.strip()
        return intent_message
    except Exception as e:
        print(f"Intent classification error: {e}")
        return "Others"

def extract_questions(text):
    # Use regex to find sentences ending with a question mark
    questions = re.findall(r'[^.!?]*\?', text)
    
    # Strip any leading/trailing whitespace from each question
    questions = [q.strip() for q in questions]
    
    # Join the questions with a space
    return ' '.join(questions)
def extract_score_and_comment(score_response):
    """
    Extracts the score and comment from the response string.
    """
    match = re.search(r'(\d+)/10\s*Comment:\s*(.*)', score_response)
    if match:
        score = int(match.group(1))  # Extracts the numerical score
        comment = match.group(2)    # Extracts the comment text
        return score, comment
    else:
        return None, None

def score_answer(question, answer):
    """
    Scores the candidate's answer to a technical question.
    """
    question = extract_questions(question)
    print("The question was:", question)
    print("\n"*4)
    print("The answer is: ", answer)
    if not question:
        return "Not APplicable"
    scoring_prompt = f"""
    You are an evaluator for a technical job interview answers. Score the candidate's answer to the question on a scale from 0 to 10.
    Provide the score only if the answer is relevant and addresses the question.
    Return "Not Applicable" if
    "1. Candidate is Greeting
    2. Asking for clarification in the asked question
    3. If the asked question is not technical."

    Below is the marking scheme:
    "4 marks for relevance, 3 for Completness, 3 for Correctness."

    After considering all these factors in the marking scheme, just return a final score out of 10.
    Question: "{question}"

    Answer: "{answer}"

    Below is Your response format:
    "?/10
    Comment: (5 to 10 words maximum)"
    Strictly follow all the instructions.
    """
    try:
        # Generate score
        score_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "user",
                    "content": scoring_prompt
                },
            ],
            temperature=1,
            # max_completion_tokens=1024,
            top_p=1,
            # stream=True,
            stop=None,
        )
        # Extract score

        # score = score_response.choices[0].message['content'].strip()
        score = score_response.choices[0].message.content.strip()
        return score
    except Exception as e:
        # print("The question was:-", question)
        # print("The answer was:-", answer)
        print(f"Scoring error: {e}")
        return "Not Applicable"

# Global variable to track interview state
# Global variable to track interview state
interview_state = {
    "messages": [],
    "question_count": 0,
    "current_question": None,
    "interview_log": [],
    "current_job_id": None,
    "current_candidate_id": None
}
def save_interview_to_csv(interview_log, candidate_name, job):
    """
    Saves the interview session to a new CSV file.

    Args:
        interview_log (list): List of dictionaries containing question, answer, score, and comment.
        candidate_name (str): Name of the candidate.
        job (str): Job position.
    """
    # Ensure the 'interviews' directory exists
    directory = "interviews"
    os.makedirs(directory, exist_ok=True)

    # Create a unique filename based on the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"{candidate_name}_{job}_{timestamp}.csv")

    # Define CSV columns
    fieldnames = ["question", "answer", "score", "comment"]

    # Write data to the CSV file
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interview_log)


def create_initial_system_message(interview_details):
    return {
        "role": "user", 
        "content": 
            f"You are a technical job interviewer for the position of {interview_details['job']}, "
            f"to assess the following skills: {interview_details['skills']}. "
            f"The candidate's name is {interview_details['candidate_name']}. "
            f"The experience/difficulty level required is: {interview_details['experience']}. "
            "Start by briefly asking about the candidate's experience. "
            "Ask exactly 2 or 3 questions for each skill listed, one by one. No compromise. "
            "Ask only one question at a time, even if the candidate requests otherwise. "
            "Do NOT repeat or stay stuck on a single question. Move on to the next question after the answer. "
            "Do NOT provide long feedback, judgment, suggestions, or commentary on the candidate's answers. "
            "Do NOT explain the correct answer."
            "JUST ask the next question."
            "Keep your questions short and to the point. "
            "Be concise, professional, and follow a real-life technical interview tone. "
            "Do NOT be overly friendly or act like it's a mock interview — this is a real interview. "
            "Strictly stay on-topic and only ask questions related to the specified skills. "
            "End the interview professionally once all questions are asked."
    }
import threading
@api_view(['POST'])
@csrf_exempt
@permission_classes([AllowAny])
def chatbot_response(request):
    """
    Modified chatbot response view with per-question scoring
    """
    try:
        # Check if this is the start of a new interview
        if 'reset' in request.data and request.data['reset']:
            job_id = request.data.get('job_id')
            candidate_id = request.data.get('candidate_id')
            print("job_id", job_id)
            print("candidate_id", candidate_id)
            if not job_id or not candidate_id:
                return Response({'error': 'job_id and candidate_id are required'}, status=400)
            
            interview_details = get_interview_details(job_id, candidate_id)
            
            # Reset interview state
            interview_state["messages"] = [create_initial_system_message(interview_details)]
            interview_state["question_count"] = 0
            interview_state["current_question"] = None
            interview_state["interview_log"] = []
            interview_state["total_score"] = 0  # Initialize total score
            interview_state["total_questions"] = 0  # Track number of scored questions
            interview_state["current_job_id"] = job_id
            interview_state["current_candidate_id"] = candidate_id
           
            initial_message = f"Hello! Nice to meet you, dear {interview_details['candidate_name']}. Can you please introduce yourself?"
            application = ApplicationTable.objects.get(
                            job_id=interview_state["current_job_id"],
                            candidate_id=interview_state["current_candidate_id"]
                        )
            application.interview_status = 'started'
            print("interview status is updated to Started")
            application.save()
            return Response({
                'response': initial_message, 
                'reset': True,
                'candidateName': interview_details['candidate_name']  # Add this line

            })
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return Response({'error': str(e)}, status=500)
    
    # Get user input
    user_input = request.data.get('message', '')
    print("Candidate Response is ", user_input)
    # Perform intent classification
    intent = get_intent(user_input)
    print(f"Detected Intent: {intent}")

    # Check if the intent is to quit the interview or max questions reached
    if intent == "Quit_interview" or interview_state["question_count"] >= 9:
        print("Yaaar ye function q chal rha hai pta ni")
        # interview_state["is_completed"] = True
        application = ApplicationTable.objects.get(
                            job_id=interview_state["current_job_id"],
                            candidate_id=interview_state["current_candidate_id"]
                        )
        application.interview_status = 'completed'
        print("interview status is updated to completed")
        application.save()
        print("Candidate has quit the interview or max questions reached.")
        # Define the face verification function
        from interview.tasks import process_post_interview

        if intent == "Quit_interview" or interview_state["question_count"] >= 9:
            print("Interview ending, triggering background tasks...")

            candidate_id = interview_state["current_candidate_id"]
            job_id = interview_state["current_job_id"]
            total_score = interview_state["total_score"]
            total_questions = interview_state["total_questions"]

            process_post_interview.delay(candidate_id, job_id, total_score, total_questions)

            response_data = {
                'response': "Thankyou for your time. The interview is now completed. Good luck with your application",
                'question_count': interview_state["question_count"],
                'intent': intent
            }
            return Response(response_data)

    # Append the user's message to the conversation history
    interview_state["messages"].append({"role": "user", "content": user_input})

    # Score the answer if there's a current question
    if interview_state["current_question"]:
        score = score_answer(interview_state["current_question"], user_input)
        # Extract score and comment
        score, comment = extract_score_and_comment(score)
        # ai_result= AI_Detetection(user_input)
        # Update total score only if it's a numeric score
        try:
            numeric_score = float(score)
            interview_state["total_score"] += numeric_score
            interview_state["total_questions"] += 1
        except (ValueError, TypeError):
            # Skip if score is "Not Applicable" or invalid
            pass
            
        print(f"Score for the answer: {score}")
        # Log the interview details
        interview_log_entry = {
            "question": interview_state["current_question"],
            "answer": user_input,
            "score": score,
            # "Detected_AI%":  f"{round(ai_result * 100, 2)}%",
            "comment": comment,
            "running_total": f"{interview_state['total_score']}/{interview_state['total_questions']*10}"
        }
        application = ApplicationTable.objects.get(
                        job_id=interview_state["current_job_id"],
                        candidate_id=interview_state["current_candidate_id"]
                    )
        application.interview_logs.append(interview_log_entry)
        print("Interview Logs has been stored in databases")
        application.save()

        interview_state["interview_log"].append(interview_log_entry)

    # Generate the assistant's response
    completion = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=interview_state["messages"],
    temperature=1,
    # max_completion_tokens=1024,
    top_p=1,
    # stream=True,
    stop=None,
    )

    # Extract the assistant's message
    assistant_message = completion.choices[0].message.content

    # Update current question if it contains a question mark
    interview_state["current_question"] = assistant_message if "?" in assistant_message else None

    # Append the assistant's response to the conversation history
    interview_state["messages"].append({"role": "assistant", "content": assistant_message})

    # Increment question count if a technical question is asked
    if "?" in assistant_message:
        interview_state["question_count"] += 1

    print('response', assistant_message,)
    # Prepare response
    response_data = {
        'response': assistant_message,
        'question_count': interview_state["question_count"],
        'intent': intent
    }
    
    return Response(response_data)


@api_view(['GET'])
@permission_classes([AllowAny])
def chatbot_page(request):
    return render(request, 'chatbot.html')



from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist
import os
import logging
import cloudinary
import cloudinary.uploader
logger = logging.getLogger(__name__)
from django.middleware.csrf import get_token
from job_posting.models import ApplicationTable # Import the ApplicationTable model
@ensure_csrf_cookie
def get_csrf_token(request):
    """
    This view sets the CSRF cookie and returns a 200 response
    """
    return JsonResponse({'csrfToken': get_token(request)})

def sanitize_filename(filename):
    # Remove invalid characters and replace with underscores
    # Windows doesn't allow: < > : " / \ | ? *
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return sanitized

@api_view(['POST']) 
@permission_classes([IsAuthenticated]) 
@ensure_csrf_cookie 
def save_frame(request): 
    try: 
        if not request.FILES.get('frame'): 
            return Response({ 
                'success': False, 
                'error': 'No frame provided' 
            }, status=400) 
 
        frame = request.FILES['frame'] 
        job_id = request.POST.get('job_id') 
        candidate_id = request.POST.get('candidate_id') 
        timestamp = request.POST.get('timestamp') 
        print("We are in save_frame function")
        print("job_id", job_id)
        print("candidate_id", candidate_id)
        # Create a safe timestamp for the filename
        safe_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') 
        
        # Create a folder path structure for Cloudinary
        folder = f'interview_frames/job_{job_id}/candidate_{candidate_id}'
        
        # Base filename 
        base_filename = f'frame_{safe_timestamp}.jpg'
        
        # Full Cloudinary filename path
        filename = f'{folder}/{base_filename}'
        
        try:
            # Upload to Cloudinary instead of local storage
            upload_result = cloudinary.uploader.upload(
                frame,
                folder=folder,
                public_id=f'frame_{safe_timestamp}',
                resource_type="image"
            )
            
            # Get the secure URL from Cloudinary
            frame_url = upload_result['secure_url']
            
            # Update ApplicationTable 
            application = ApplicationTable.objects.get( 
                job_id=job_id, 
                candidate_id=candidate_id 
            ) 
             
            # Initialize or update interview_frames 
            current_frames = application.interview_frames
            if not isinstance(current_frames, list): 
                current_frames = [] 
             
            frame_data = { 
                'url': frame_url, 
                'timestamp': timestamp, 
                'filename': filename  # Store the Cloudinary path
            } 
             
            current_frames.append(frame_data) 
            application.interview_frames = current_frames 
            application.save() 
             
            return Response({ 
                'success': True, 
                'url': frame_url, 
                'frame_data': frame_data 
            }) 
             
        except Exception as e: 
            # Log the error for debugging
            print(f"Error uploading to Cloudinary: {str(e)}")
            return Response({
                'success': False,
                'error': f'Failed to upload image: {str(e)}'
            }, status=500)
    except Exception as e:
        logger.error(f"Error in save_frame: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from authentication.models import CandidateTable
from django.shortcuts import get_object_or_404

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_current_candidate(request):
    """
    Endpoint to get the current logged-in candidate's details
    """
    try:
        candidate = get_object_or_404(CandidateTable, candidate=request.user)
        profile_image_url = candidate.profile_image.url if candidate.profile_image else None

        return Response({
            'id': candidate.id,
            'name': candidate.candidate_name,
            'email': candidate.candidate_email,
            'profile': profile_image_url,
        })
    except Exception as e:
        return Response({'error': str(e)}, status=400)
    



from rest_framework.response import Response
import requests
from django.conf import settings

def verify_interview_frames(candidate_id, job_id):
    """
    Call face verification endpoint and get results for all frames
    """
    try:
        # Get interview details including frames and profile image
        interview_details = get_interview_details(job_id, candidate_id)
        print("interview_details",interview_details)
        # Prepare data for face verification
        verification_data = {
            "profile_image": (
                    interview_details.get('profile_image').url
                    if interview_details.get('profile_image') else None
                ),

                        "frames": [
                {"url": frame["url"].url if hasattr(frame["url"], "url") else frame["url"]}
                for frame in interview_details.get('interview_frames', [])
            ],

        }

        # Using properly formatted URL
        verification_url = f"{settings.BASE_URL}/api/confidence_prediction/verify-face/cheat/"
        
        # Call face verification endpoint
        response = requests.post(
            verification_url,
            json=verification_data,
        )

        # print("response",response)
        
        if response.status_code == 200:
            return response.json()
        return None

    except Exception as e:
        logger.error(f"Error in frame verification: {e}")
        return None
    

from django.conf import settings

def confidence_prediction(candidate_id, job_id):
    try:
        import json, traceback, requests, re, os
        from django.conf import settings

        interview_details = get_interview_details(job_id, candidate_id)
        if not interview_details:
            print("No interview details found")
            return None

        frames = interview_details.get('interview_frames', [])
        if not frames:
            print("No frames found in interview details")
            return None

        base_url = os.getenv("BASE_URL")
        confidence_url = f"{base_url}/api/confidence_prediction/analyze-confidence/"
        print(f"Confidence endpoint: {confidence_url}")

        frame_data = []
        for frame in frames:
            url = frame.get("url", "").rstrip("';\"")  # Clean trailing chars
            if 'cloudinary.com' in url:
                full_url = url
            else:
                domain = settings.BASE_URL.rstrip('/')
                relative_url = url.lstrip('/')
                full_url = f"{domain}/{relative_url}"
            frame_data.append({"url": full_url})

        confidence_data = {"frames": frame_data}
        print(f"Sending {len(frame_data)} frames...")

        response = requests.post(confidence_url, json=confidence_data, timeout=120)

        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                print("Invalid JSON in response")
                return None
        else:
            print(f"Confidence prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"Exception in confidence_prediction: {e}")
        traceback.print_exc()
        return None

import tempfile

@api_view(['POST'])
def transcribe_audio(request):
    """
    Endpoint to receive audio file and transcribe it using Groq's Whisper model
    """
    if 'audio' not in request.FILES:
        return Response({'error': 'No audio file provided'}, status=400)
    
    audio_file = request.FILES['audio']
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
        for chunk in audio_file.chunks():
            temp_file.write(chunk)
        temp_file_path = temp_file.name
    
    try:
        # Transcribe using Groq's Whisper model
        with open(temp_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_file_path, file.read()),
                model="distil-whisper-large-v3-en",
                response_format="verbose_json",
            )
        
        # Extract transcribed text
        transcribed_text = transcription.text
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return Response({
            'transcription': transcribed_text,
            'success': True
        })
    
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        print(f"Transcription error: {e}")
        return Response({
            'error': str(e),
            'success': False
        }, status=500)
