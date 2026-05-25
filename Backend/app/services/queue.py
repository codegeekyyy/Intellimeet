# When audio is uploaded, we will eventually dispatch it to a job queue. To make sure the routers compile and run now without errors
import logging

logger = logging.getLogger("intellimeet")

async def dispatch_audio_job(session_id: str, audio_path: str) -> str:
    """
    Temporary mock job dispatcher. 
    Will be replaced with real Redis BullMQ integration in Phase 4.
    """
    logger.info(f"[Queue Stub] Dispatched audio processing job for session {session_id} using audio {audio_path}")
    return "mock_job_id_12345"