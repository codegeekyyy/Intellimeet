import asyncio
import json
import logging
import sys
from redis.asyncio import Redis

# Ensure parent directory is in python path
sys.path.append(sys.path[0] + "/../..")

from app.config import settings
from app.services.queue import dispatch_llm_job, set_job_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Audio Worker: %(message)s")
logger = logging.getLogger("audio_worker")

from workers.audio_processor.pipeline import run_audio_pipeline

async def main():
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("Audio processor worker started and listening to 'bull:audio:waiting'...")
    
    try:
        while True:
            # BRPOP blocks until a job becomes available in the list
            result = await redis.brpop("bull:audio:waiting", timeout=0)
            if result:
                _, payload_str = result
                job = json.loads(payload_str)
                await run_audio_pipeline(job)
    except asyncio.CancelledError:
        logger.info("Audio worker shutting down gracefully...")
    finally:
        await redis.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Audio worker stopped by user.")


# fastapi->audio dispatched to redis queue->audio worker ->llm redis queue->llm worker->summarization stored in the storage-> fastapi gets job done-> UI is updated.