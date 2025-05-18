# interview/tasks.py

from celery import shared_task
from .models import ApplicationTable
from interview.views import verify_interview_frames, confidence_prediction  # Import your functions

@shared_task
def process_post_interview(candidate_id, job_id, total_score, total_questions):
    print("Running Facing Verification Model...!!😊!!")

    try:
        application = ApplicationTable.objects.get(job_id=job_id, candidate_id=candidate_id)
        application.interview_status = 'completed'
        application.save()
        print("Interview status set to completed.")
    except Exception as e:
        print(f"Error fetching/updating application: {e}")
        return

    full_verification_response = verify_interview_frames(candidate_id, job_id)
    verification_results_list = full_verification_response.get("verification_results", [])
    verification_summary = full_verification_response.get("summary", {})

    confidence_results = confidence_prediction(candidate_id, job_id)
    confidence_results_score = confidence_results.get("final_score")

    try:
        application.confidence_score = confidence_results_score
        if total_questions > 0:
            marks_string = f"{round(total_score, 1)}/{total_questions * 10}"
            application.marks = marks_string
        application.face_verification_result = verification_results_list

        verification_rate = verification_summary.get("verification_rate", 0)
        application.flag_status = verification_rate <= 80
        application.save()
        print("Saved all post-interview results.")

    except Exception as e:
        print(f"Error saving results to DB: {e}")
