import asyncio
import json
import logging
import sys
import os

# Put Backend in Python path
sys.path.append(sys.path[0] + "/../..")

# Mock optional packages for SpeechBrain to prevent importlib find_spec failures during lightning stack inspects
from unittest.mock import MagicMock
import importlib.machinery
def mock_module(name):
    mock = MagicMock()
    mock.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = mock

mock_module('k2')
mock_module('flair')
mock_module('numba')

# Monkeypatch SpeechBrain LazyModule to ignore import errors during dynamic inspect checks
try:
    import speechbrain.utils.importutils as sb_import
    old_ensure = sb_import.LazyModule.ensure_module
    def patched_ensure(self, stacklevel=1):
        try:
            return old_ensure(self, stacklevel)
        except Exception:
            return MagicMock()
    sb_import.LazyModule.ensure_module = patched_ensure
except Exception:
    pass

from redis.asyncio import Redis
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