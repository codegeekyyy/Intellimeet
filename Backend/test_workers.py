import asyncio
import json
import os
import uuid
from pydub.generators import Sine
from sqlalchemy import select, delete

from app.database import AsyncSessionLocal
from app.models.user import User
from app.models.session import MeetingSession
from app.models.transcript import Transcript
from app.services.queue import dispatch_audio_job, get_job_progress, redis_client
from workers.audio_processor.worker import main as run_audio_worker
from workers.llm_processor.worker import main as run_llm_worker

async def test_full_pipeline():
    print("\n--- Testing Redis Queue & Worker Pipeline (Async) ---")

    # Generate identifiers
    test_user_id = uuid.uuid4()
    test_session_id = uuid.uuid4()
    session_id_str = str(test_session_id)
    audio_path = f"test_user_123/{test_session_id}.wav"
    
    # 1. Create a dummy test file in the uploads directory
    test_file_dir = os.path.join("uploads", "test_user_123")
    os.makedirs(test_file_dir, exist_ok=True)
    test_file_path = os.path.join("uploads", audio_path)
    
    print(f"Generating 5-second test WAV at {test_file_path}...")
    tone = Sine(440).to_audio_segment(duration=5000)
    tone.export(test_file_path, format="wav")

    # 2. Setup database records (User & Session)
    async with AsyncSessionLocal() as db:
        # Get or create dummy user
        result = await db.execute(select(User).where(User.email == "test_worker_user@test.com"))
        user = result.scalars().first()
        if not user:
            user = User(
                id=test_user_id,
                email="test_worker_user@test.com",
                username="test_worker_user",
                hashed_password="dummy_password",
                is_active=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        else:
            test_user_id = user.id

        # Create session record
        session = MeetingSession(
            id=test_session_id,
            user_id=test_user_id,
            title="Test Worker Meeting",
            status="queued",
            audio_path=audio_path
        )
        db.add(session)
        await db.commit()
        print(f"Created database session record {session_id_str}")

    # Start workers in the background
    audio_worker_task = asyncio.create_task(run_audio_worker())
    llm_worker_task = asyncio.create_task(run_llm_worker())

    # Wait a second for workers to initialize
    await asyncio.sleep(1)

    # 3. Dispatch job
    print(f"Dispatching audio job for session {session_id_str}...")
    job_id = await dispatch_audio_job(session_id_str, audio_path)
    
    # 4. Poll progress in Redis to verify workers are processing
    print("Polling job progress from Redis:")
    success = False
    
    for attempt in range(40):  # Poll up to 40 times (about 32 seconds total)
        await asyncio.sleep(0.8)
        progress_info = await get_job_progress(session_id_str)
        progress = progress_info["progress"]
        status = progress_info["status"]
        
        print(f"  Attempt {attempt + 1:02d}: Progress = {progress}% | Status = {status}")
        
        # When audio worker is done it dispatches to LLM worker, which sets status="complete" / progress=100
        # Wait, since the LLM worker currently is a mock worker, it will mock-complete the summary job
        if status == "complete" and progress == 100:
            print("Worker pipeline completed successfully! [OK]")
            success = True
            break
        elif status == "failed":
            print("Pipeline status updated to FAILED in Redis [FAIL]")
            break
            
    # Clean up background worker tasks
    audio_worker_task.cancel()
    llm_worker_task.cancel()
    
    # Ensure they shut down cleanly
    try:
        await asyncio.gather(audio_worker_task, llm_worker_task, return_exceptions=True)
    except Exception:
        pass

    # 5. Clean up Database records & files
    print("Cleaning up database records & files...")
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Transcript).where(Transcript.session_id == test_session_id))
        await db.execute(delete(MeetingSession).where(MeetingSession.id == test_session_id))
        await db.commit()

    if os.path.exists(test_file_path):
        os.remove(test_file_path)

    # Clean up Redis keys used for testing
    await redis_client.delete(f"job:{job_id}:progress")
    await redis_client.delete(f"session:{session_id_str}:status")
    await redis_client.close()

    assert success, "Worker pipeline did not complete or reach 100%!"
    print("--- Redis Queue & Worker Pipeline Tests Passed! ---")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
