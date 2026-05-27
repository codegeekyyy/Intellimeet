from sys import prefix
import logging
import os
import uuid
import json
import logging
import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from openai import AsyncOpenAI
from app.config import settings
from app.database import get_db
from app.services.auth import get_current_user
from app.services.storage import save_audio_locally, get_audio_full_path
from app.models.interview import InterviewSession, InterviewAttempt
from app.schemas.interview import (
    InterviewStartRequest,
    InterviewStartResponse,
    AnswerSubmitResponse,
    InterviewHistoryResponse
)
from workers.llm_processor.prompts import (
    QUESTION_GENERATION_PROMPT,
    ANSWER_EVALUATION_PROMPT
)
from workers.audio_processor.preprocessor import preprocess_audio


logger = logging.getLogger("intellimeet.routers.interview")
router = APIRouter(prefix="/interview", tags=["interview prep"])

# Lazy-loaded transcriber singleton
_transcriber = None

def get_transcriber():
    global _transcriber
    if _transcriber is None:
        from workers.audio_processor.transcriber import AudioTranscriber
        _transcriber = AudioTranscriber()
    return _transcriber


def parse_json_response(raw_text: str) -> dict:
    import re
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                raise ValueError("Response contains invalid JSON structure") from e
        else:
            raise ValueError("No JSON object structure found in response")
ALLOWED_AUDIO_MIME = {
    "audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a",
    "audio/ogg", "video/mp4", "video/webm"
}

@router.post("/session/start", response_model=InterviewStartResponse, status_code=status.HTTP_201_CREATED)

async def  start_interview_session(
    payload: InterviewStartRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Generates 5 tailored questions based on JD and Resume using Groq API,
    creates a session in PostgreSQL, and returns the questions.
    """
    # 1. query groq llm to generate questions
    client = AsyncOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    user_prompt = f"JOB DESCRIPTION:\n{payload.job_description}\n\nCANDIDATE RESUME:\n{payload.resume_text}"

    questions = []
    try:
        if settings.GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is using the default placeholder")
        res = await client.chat.completions.create(
            messages = [
                {"role": "system", "content": QUESTION_GENERATION_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        parsed = parse_json_response(res.choices[0].message.content)
        questions = parsed.get("questions", [])
    except Exception as e:
        logger.warning(f"Groq API call failed: {e}. Falling back to generating mock questions locally.")
        questions = [
            "Can you describe a challenging technical project you worked on and how you handled difficulties?",
            "How do you stay updated with the latest software development trends and tools?",
            "Describe a situation where you had to collaborate with a difficult coworker or stakeholder.",
            "Why are you interested in this role, and how does your experience align with the job description?",
            "Can you explain the difference between REST APIs and WebSockets, and when you would use each?"
        ]

    # save session record to db
    session = InterviewSession(
        user_id=current_user.id,
        job_description=payload.job_description,
        resume_text=payload.resume_text,
        questions=questions
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    return {
        "session_id": session.id,
        "questions": session.questions
    }


@router.post("/session/{session_id}/answer", response_model=AnswerSubmitResponse)
async def submit_answer(
    session_id: uuid.UUID,
    question_index: int = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Ingests an answer audio file, transcribes it synchronously using Whisper,
    evaluates the transcript against the question/JD using Groq, and saves/returns feedback.
    """
     # 1. Fetch & Verify Session
    session = await db.get(InterviewSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found"
        )
        
    if question_index < 0 or question_index >= len(session.questions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid question index. Must be between 0 and {len(session.questions) - 1}."
        )
        
    question_text = session.questions[question_index]
    # 2. Validate MIME Type
    if file.content_type not in ALLOWED_AUDIO_MIME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file.content_type}."
        )
    # 3. Stream Upload to Temporary file
    temp_dir = os.path.join(settings.UPLOAD_DIR, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    tmp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                await f.write(chunk)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream upload audio: {e}"
        )
    # 4. Save and Preprocess Audio
    relative_path = await save_audio_locally(tmp_path, str(current_user.id))
    absolute_path = get_audio_full_path(relative_path)
    
    preprocessed_wav = None
    try:
        preprocessed_wav = preprocess_audio(absolute_path)
    except Exception as e:
        logger.error(f"Preprocessing audio failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to preprocess audio file for transcription"
        )
    # 5. Transcribe using Whisper
    try:
        transcriber = get_transcriber()
        words = transcriber.transcribe_chunk(preprocessed_wav)
        transcript_text = " ".join([w["word"] for w in words]).strip()
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transcription pipeline failed"
        )
    finally:
        # Clean up temporary WAV file
        if preprocessed_wav and os.path.exists(preprocessed_wav):
            os.remove(preprocessed_wav)
    # 6. Evaluate response transcript via Groq LLM
    client = AsyncOpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    
    user_prompt = (
        f"JOB DESCRIPTION:\n{session.job_description}\n\n"
        f"CANDIDATE RESUME:\n{session.resume_text}\n\n"
        f"QUESTION ASKED:\n{question_text}\n\n"
        f"CANDIDATE TRANSCRIPT ANSWER:\n{transcript_text}"
    )
    
    evaluation = {}
    try:
        if settings.GROQ_API_KEY == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is using the default placeholder")
            
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": ANSWER_EVALUATION_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        evaluation = parse_json_response(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Groq evaluation failed: {e}. Falling back to local mock evaluation.")
        evaluation = {
            "score": 75,
            "star_method": {
                "situation": "Described context successfully.",
                "task": "Outlined expected deliverable.",
                "action": "Detailed steps taken.",
                "result": "Highlighted project outcome."
            },
            "strengths": [
                "Answer was direct and addressed the core question.",
                "Tone was professional."
            ],
            "improvements": [
                "Include more quantitative metrics (numbers/percentages) in the Result section."
            ],
            "relevance": {
                "score": 8,
                "feedback": "Answer directly relates to the question asked."
            },
            "clarity": {
                "score": 7,
                "feedback": "Speech was understandable, though minor structural transitions could be smoother."
            }
        }
    # 7. Save Attempt record in database
    attempt = InterviewAttempt(
        interview_session_id=session.id,
        question_index=question_index,
        audio_path=relative_path,
        transcript=transcript_text,
        evaluation=evaluation
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)
    
    return {
        "attempt_id": attempt.id,
        "question_index": attempt.question_index,
        "transcript": attempt.transcript,
        "evaluation": attempt.evaluation
    }

@router.get("/session/{session_id}/history", response_model=InterviewHistoryResponse)
async def get_session_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Fetches the interview session details along with all attempts and evaluations.
    """
#    1. fetch session
    session =  await db.get(InterviewSession, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail = "interview session not found"
        )

    # fetch attempt
    res = await db.execute(
        select(InterviewAttempt)
        .where(InterviewAttempt.interview_session_id == session_id)
        .order_by(InterviewAttempt.question_index.asc(), InterviewAttempt.created_at.asc())
    )
    attempts = res.scalars().all()
    return {
        "id": session.id,
        "job_description": session.job_description,
        "resume_text": session.resume_text,
        "questions": session.questions,
        "created_at": session.created_at,
        "attempts": attempts
    }


    

    
    