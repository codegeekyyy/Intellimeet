# workers/llm_processor/worker.py

import os
import sys
import json
import logging
import asyncio
import uuid
from redis.asyncio import Redis
from openai import AsyncOpenAI
from sqlalchemy import select, delete

# Ensure parent directory is in python path
sys.path.append(sys.path[0] + "/../..")

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.session import MeetingSession
from app.models.transcript import Transcript
from app.models.summary import Summary
from app.services.queue import set_job_progress
from workers.llm_processor.prompts import MEETING_SYSTEM_PROMPT, build_meeting_prompt
from workers.llm_processor.parser import parse_summary_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LLM Worker: %(message)s")
logger = logging.getLogger("llm_worker")

async def run_llm_pipeline(job: dict):
    job_id = job["id"]
    session_id = job["session_id"]
    logger.info(f"Processing LLM summary for job {job_id}, session {session_id}...")

    try:
        # Step 1: Fetch Transcript from Database
        await set_job_progress(job_id, 20, "fetching_transcript")
        async with AsyncSessionLocal() as db:
            # Query session
            session = await db.get(MeetingSession, uuid.UUID(session_id))
            if not session:
                raise ValueError(f"Session {session_id} not found in database")

            # Query transcript
            result = await db.execute(select(Transcript).where(Transcript.session_id == session.id))
            transcript_record = result.scalars().first()
            if not transcript_record or not transcript_record.segments:
                logger.warning(f"No transcript segments found for session {session_id}. Saving empty summary.")
                # Save empty summary and set status to complete
                await save_empty_summary(db, session)
                await set_job_progress(job_id, 100, "complete")
                return

            segments = transcript_record.segments

        # Step 2: Format Transcript & Query Groq API
        await set_job_progress(job_id, 50, "generating_summary")
        user_prompt = build_meeting_prompt(segments)
        
        # Initialize OpenAI client with Groq settings
        client = AsyncOpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Try primary model and fall back to secondary models if needed
        models_to_try = [
            "llama-3.1-70b-versatile",
            "llama-3.3-70b-versatile",
            "llama3-70b-8192",
            "llama-3.1-8b-instant"
        ]
        
        raw_content = None
        tokens_used = "0"
        model_used = None
        
        for model in models_to_try:
            logger.info(f"Querying Groq API using model '{model}'...")
            try:
                if settings.GROQ_API_KEY == "your_groq_api_key_here":
                    raise ValueError("GROQ_API_KEY is placeholder")
                response = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": MEETING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model, 
                    temperature=0.2,
                    max_tokens=4096,
                    response_format={"type": "json_object"}
                )
                raw_content = response.choices[0].message.content
                tokens_used = f"in:{response.usage.prompt_tokens},out:{response.usage.completion_tokens}"
                model_used = model
                logger.info(f"Received LLM response from '{model}'. Tokens used: {tokens_used}")
                break
            except Exception as e:
                logger.warning(f"Failed to query Groq model '{model}': {e}. Trying next model...")
                continue
                
        if raw_content is None:
            logger.warning("All configured Groq models failed (or GROQ_API_KEY is not set). Falling back to mock summary generation to complete the pipeline.")
            model_used = "mock-fallback"
            tokens_used = "mock:in:0,out:0"
            
            # Build a mock summary locally
            full_text = " ".join([seg.get("text", "") for seg in segments])
            if not full_text.strip():
                full_text = "No text content detected in segments."
                
            overview = f"Locally generated summary (Groq API unavailable). Transcript segment count: {len(segments)}. Content summary: {full_text[:300]}..."
            
            decisions = []
            action_items = []
            for seg in segments:
                text = seg.get("text", "")
                speaker = seg.get("speaker", "Unknown")
                text_lower = text.lower()
                if any(x in text_lower for x in ["decided", "agree", "choose", "chose", "settle"]):
                    decisions.append(f"{speaker} decided: {text.strip()}")
                if any(x in text_lower for x in ["will", "todo", "need to", "should", "must", "action"]):
                    action_items.append(f"Action for {speaker}: {text.strip()}")
                    
            if not decisions:
                decisions = ["No explicit decisions were identified in the transcript."]
            if not action_items:
                action_items = ["No action items were identified in the transcript."]
                
            raw_content = json.dumps({
                "overview": overview,
                "decisions": decisions[:5],
                "action_items": action_items[:5],
                "open_questions": ["What are the next steps for implementation?"],
                "sentiment": {
                    "overall": "positive" if "great" in full_text.lower() or "good" in full_text.lower() else "neutral",
                    "notes": "Generated locally due to Groq API key issue or rate limit."
                }
            })

        # Step 3: Parse and Validate Output JSON
        await set_job_progress(job_id, 80, "parsing_summary")
        parsed_summary = parse_summary_response(raw_content)

        # Step 4: Save Summary to Database & Complete Session
        await set_job_progress(job_id, 90, "saving_summary")
        async with AsyncSessionLocal() as db:
            # Re-fetch session
            session = await db.get(MeetingSession, uuid.UUID(session_id))
            if session:
                # Clean up any existing summaries for this session
                await db.execute(delete(Summary).where(Summary.session_id == session.id))
                
                summary_record = Summary(
                    session_id=session.id,
                    overview=parsed_summary["overview"],
                    decisions=parsed_summary["decisions"],
                    action_items=parsed_summary["action_items"],
                    open_questions=parsed_summary["open_questions"],
                    sentiment=parsed_summary["sentiment"],
                    model_used=model_used or "unknown",
                    tokens_used=tokens_used
                )
                db.add(summary_record)
                
                # Set session status to complete
                session.status = "complete"
                await db.commit()
                logger.info(f"Saved meeting summary for session {session_id} to database. Status set to complete.")
            else:
                raise ValueError(f"Session {session_id} disappeared during processing")

        await set_job_progress(job_id, 100, "complete")
        
    except Exception as e:
        logger.error(f"Error in LLM pipeline for session {session_id}: {e}", exc_info=True)
        try:
            async with AsyncSessionLocal() as db:
                session = await db.get(MeetingSession, uuid.UUID(session_id))
                if session:
                    session.status = "failed"
                    await db.commit()
            await set_job_progress(job_id, 100, "failed")
        except Exception as db_err:
            logger.error(f"Failed to update session failure status in DB: {db_err}")
        raise e

async def save_empty_summary(db, session):
    """Saves an empty summary layout to database on empty transcripts."""
    await db.execute(delete(Summary).where(Summary.session_id == session.id))
    summary_record = Summary(
        session_id=session.id,
        overview="Empty transcript. No overview generated.",
        decisions=[],
        action_items=[],
        open_questions=[],
        sentiment={"overall": "neutral", "notes": "No speech detected."},
        model_used="none",
        tokens_used="0"
    )
    db.add(summary_record)
    session.status = "complete"
    await db.commit()

async def main():
    redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    logger.info("LLM summary worker started and listening to 'bull:llm:waiting'...")
    
    try:
        while True:
            # BRPOP blocks until a job becomes available in the queue
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
