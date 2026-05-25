import json
import logging
from redis.asyncio import Redis
from app.config import settings

logger = logging.getLogger("intellimeet")

# Setup Redis connection pool
redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)

async def dispatch_audio_job(session_id: str, audio_path: str) -> str:
    """
    Dispatches a job to the Redis queue for the audio worker.
    """
    job_id = f"job_{session_id}"
    job_payload = {
        "id": job_id,
        "session_id": session_id,
        "audio_path": audio_path
    }
    
    # LPUSH into the audio worker queue
    await redis_client.lpush("bull:audio:waiting", json.dumps(job_payload))
    
    # Initialize progress to 0%
    await redis_client.set(f"job:{job_id}:progress", 0, ex=86400)  # expires in 1 day
    logger.info(f"Dispatched audio job {job_id} to queue 'bull:audio:waiting'")
    
    return job_id

async def dispatch_llm_job(session_id: str) -> str:
    """
    Dispatches a job to the Redis queue for the LLM worker.
    """
    job_id = f"job_llm_{session_id}"
    job_payload = {
        "id": job_id,
        "session_id": session_id
    }
    
    # LPUSH into the LLM worker queue
    await redis_client.lpush("bull:llm:waiting", json.dumps(job_payload))
    logger.info(f"Dispatched LLM job {job_id} to queue 'bull:llm:waiting'")
    
    return job_id

async def set_job_progress(job_id: str, progress: int, status: str) -> None:
    """Update progress percentage (0-100) and status in Redis."""
    session_id = job_id.replace("job_llm_", "", 1).replace("job_", "", 1)
    await redis_client.set(f"job:{job_id}:progress", progress, ex=86400)
    await redis_client.set(f"session:{session_id}:status", status, ex=86400)

async def get_job_progress(session_id: str) -> dict:
    """Get the current progress value and status from Redis."""
    job_id = f"job_{session_id}"
    progress = await redis_client.get(f"job:{job_id}:progress")
    status_val = await redis_client.get(f"session:{session_id}:status")
    
    return {
        "progress": int(progress) if progress else 0,
        "status": status_val or "queued"
    }