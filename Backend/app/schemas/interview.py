from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import List, Dict, Any, Optional

class InterviewStartRequest(BaseModel):
    job_description: str
    resume_text: str

class InterviewStartResponse(BaseModel):
    session_id: UUID
    questions: List[str]


class AnswerSubmitResponse(BaseModel):
    attempt_id: UUID
    question_index : int
    transcript : str
    evaluation : Dict[str, Any]


class AttemptDetail(BaseModel):
    id: UUID
    question_index: int
    audio_path: Optional[str] = None
    transcript: Optional[str] = None
    evaluation: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class InterviewHistoryResponse(BaseModel):
    id: UUID
    job_description: str
    resume_text: str
    questions: List[str]
    created_at: datetime
    attempts: List[AttemptDetail]