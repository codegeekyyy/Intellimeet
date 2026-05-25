import os
import uuid
import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.session import MeetingSession
from app.services.auth import get_current_user
from app.services.storage import save_audio_locally
from app.services.queue import dispatch_audio_job

router = APIRouter(prefix="/audio", tags=["Audio Ingestion"])

ALLOWED_MIME = {
    "audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a",
    "audio/ogg", "video/mp4", "video/webm"
}

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_meeting(
    file: UploadFile = File(...),
    title: str = Form("Untitled Meeting"),
    db: AsyncSession = Depends(get_db),
    user = Depends(get_current_user)
):
    # 1. Validate MIME type
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {file.content_type}. Allowed: WAV, MP3, M4A, OGG, MP4, WEBM."
        )

    # Ensure local temp upload folder exists
    temp_dir = os.path.join(settings.UPLOAD_DIR, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Stream upload to a temporary file in chunks (avoids RAM bloop)
    ext = os.path.splitext(file.filename)[1]
    tmp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    
    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."
                    )
                await f.write(chunk)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    # 3. Move from temp to permanent user storage
    audio_path = await save_audio_locally(tmp_path, str(user.id))

    # 4. Save session record in PostgreSQL database
    session = MeetingSession(
        user_id=user.id,
        title=title,
        audio_path=audio_path,
        status="queued"
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)

    # 5. Dispatch job to worker pipeline (represented by our queue service)
    job_id = await dispatch_audio_job(str(session.id), audio_path)

    return {
        "session_id": str(session.id),
        "job_id": job_id,
        "status": "queued"
    }
