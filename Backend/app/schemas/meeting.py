from datetime import datetime
from uuid import UUID
from typing import Any
from pydantic import BaseModel

class MeetingResponse(BaseModel):
    id: UUID
    title: str
    status: str
    duration_s: float | None =  None
    created_at: datetime


    class Config:
        from_attributes = True

class SegmentSchema(BaseModel):
    speaker: str
    start: float
    end: float
    text: str
    confidence: float | None = None

class TranscriptResponse(BaseModel):
    segments: list[SegmentSchema]
    raw_wer: float | None = None
    class Config:
        from_attributes = True


class SummaryResponse(BaseModel):
    overview: str | None = None
    decisions: list[Any] | None = None
    action_items: list[Any] | None = None
    open_questions: list[Any] | None = None
    sentiment: dict[str, Any] | None = None
    class Config:
        from_attributes = True


class MeetingDetailResponse(BaseModel):
    id: UUID
    title: str
    status: str
    duration_s: float | None = None
    created_at: datetime
    transcript: TranscriptResponse | None = None
    summary: SummaryResponse | None = None
    class Config:
        from_attributes = True