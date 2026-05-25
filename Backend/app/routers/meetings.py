from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from uuid import UUID

from app.database import get_db
from app.models.session import MeetingSession
from app.models.transcript import Transcript
from app.models.summary import Summary
from app.schemas.meeting import MeetingResponse, MeetingDetailResponse
from app.services.auth import get_current_user
from app.services.storage import delete_audio_file

router = APIRouter(prefix="/meetings", tags=["Meetings Management"])

@router.get("/", response_model=list[MeetingResponse])
async def list_meetings(
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    # Fetch all meetings for current user sorted by date descending
    result = await db.execute(
        select(MeetingSession)
        .where(MeetingSession.user_id == user.id)
        .order_by(MeetingSession.created_at.desc())
    )
    return result.scalars().all()

@router.get("/{session_id}", response_model=MeetingDetailResponse)
async def get_meeting_detail(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    # Fetch session
    session_result = await db.execute(
        select(MeetingSession)
        .where(MeetingSession.id == session_id, MeetingSession.user_id == user.id)
    )
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting session not found"
        )

    # Fetch related transcript
    transcript_result = await db.execute(
        select(Transcript).where(Transcript.session_id == session_id)
    )
    transcript = transcript_result.scalar_one_or_none()

    # Fetch related summary
    summary_result = await db.execute(
        select(Summary).where(Summary.session_id == session_id)
    )
    summary = summary_result.scalar_one_or_none()

    # Return combined detail object
    return {
        "id": session.id,
        "title": session.title,
        "status": session.status,
        "duration_s": session.duration_s,
        "created_at": session.created_at,
        "transcript": transcript,
        "summary": summary
    }

@router.delete("/{session_id}", status_code=status.HTTP_200_OK)
async def delete_meeting(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    # Fetch session
    result = await db.execute(
        select(MeetingSession)
        .where(MeetingSession.id == session_id, MeetingSession.user_id == user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meeting session not found"
        )

    # Delete physical audio file from disk
    if session.audio_path:
        delete_audio_file(session.audio_path)

    # Remove session from database (Cascades will delete transcript & summary)
    await db.delete(session)
    await db.commit()

    return {"detail": "Meeting session deleted successfully"}
