import asyncio
import json
import logging
import sys
from redis.asyncio import Redis

sys.path.append(sys.path[0] + "/../..")

from app.config import settings
from app.services.queue import set_job_progress

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LLM Worker: %(message)s")
logger = logging.getLogger("llm_worker")

async def run_llm_pipeline(job: dict):
    session_id = job["session_id"]
    job_id = job["id"]
    audio_job_id = f"job_{session_id}"
    logger.info(f"Processing LLM job {job_id} for session {session_id}...")

    # Simulated LLM steps
    await set_job_progress(audio_job_id, 95, "summarizing")
    await asyncio.sleep(0.3)  # Shortened delay for test speed

    await set_job_progress(audio_job_id, 100, "complete")
    logger.info(f"LLM pipeline complete for session {session_id}. Summary saved [Complete].")

async def main():
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("LLM processor worker started and listening to 'bull:llm:waiting'...")
    
    try:
        while True:
            result = await redis.brpop("bull:llm:waiting", timeout=0)
            if result:
                _, payload_str = result
                job = json.loads(payload_str)
                await run_llm_pipeline(job)
    except asyncio.CancelledError:
        logger.info("LLM worker shutting down gracefully...")
    finally:
        await redis.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("LLM worker stopped by user.")
