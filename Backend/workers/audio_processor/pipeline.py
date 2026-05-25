import os
import uuid
import logging
import shutil
from pydub import AudioSegment
from app.database import AsyncSessionLocal
from app.models.session import MeetingSession
from app.models.transcript import Transcript
from app.services.storage import get_audio_full_path
from app.services.queue import dispatch_llm_job, set_job_progress

from workers.audio_processor.preprocessor import preprocess_audio, chunk_audio
from workers.audio_processor.transcriber import AudioTranscriber
from workers.audio_processor.diarizer import AudioDiarizer, align_words_to_speakers, group_into_segments
from workers.audio_processor.separator import AudioSeparator

logger = logging.getLogger("audio_worker.pipeline")

# Lazy-loaded singletons to prevent multiple model loads and conserve RAM/VRAM
transcriber_model = None
diarizer_model = None
separator_model = None

def get_transcriber():
    global transcriber_model
    if transcriber_model is None:
        transcriber_model = AudioTranscriber()
    return transcriber_model

def get_diarizer():
    global diarizer_model
    if diarizer_model is None:
        diarizer_model = AudioDiarizer()
    return diarizer_model

def get_separator():
    global separator_model
    if separator_model is None:
        separator_model = AudioSeparator()
    return separator_model

def compute_overlap_ratio(turns: list[dict], duration_s: float) -> float:
    """
    Compute overlap ratio from diarization turns.
    Overlap is defined as time ranges where more than one speaker is speaking simultaneously.
    """
    if not turns or duration_s <= 0:
        return 0.0
        
    events = []
    for turn in turns:
        events.append((turn["start"], 1, turn["speaker"]))
        events.append((turn["end"], -1, turn["speaker"]))
        
    # Sort events by time. End events (-1) come before start events (+1) to avoid boundary glitches.
    events.sort(key=lambda x: (x[0], x[1]))
    
    overlap_duration = 0.0
    active_speakers = set()
    prev_time = 0.0
    
    for time, event_type, speaker in events:
        if len(active_speakers) > 1 and time > prev_time:
            overlap_duration += (time - prev_time)
            
        if event_type == 1:
            active_speakers.add(speaker)
        else:
            active_speakers.discard(speaker)
            
        prev_time = time
        
    return overlap_duration / duration_s

async def run_audio_pipeline(job: dict):
    job_id = job["id"]
    session_id = job["session_id"]
    relative_audio_path = job["audio_path"]
    
    logger.info(f"Starting audio pipeline for job {job_id}, session {session_id}...")
    
    input_path = get_audio_full_path(relative_audio_path)
    preprocessed_wav_path = None
    separated_tracks = []
    used_separation = False
    
    try:
        # Step 1: Downloading / Accessing file
        await set_job_progress(job_id, 5, "downloading")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Uploaded audio file not found on disk: {input_path}")
            
        # Step 2: Preprocessing (conversion & volume normalization)
        await set_job_progress(job_id, 15, "preprocessing")
        preprocessed_wav_path = preprocess_audio(input_path)
        
        # Load preprocessed audio to measure duration
        audio_segment = AudioSegment.from_wav(preprocessed_wav_path)
        duration_s = len(audio_segment) / 1000.0
        logger.info(f"Audio duration: {duration_s:.2f} seconds.")
        
        # Step 3: Diarizing (speaker identification)
        await set_job_progress(job_id, 30, "diarizing")
        diarizer = get_diarizer()
        turns = diarizer.diarize(preprocessed_wav_path)
        
        # Calculate overlap ratio
        overlap_ratio = compute_overlap_ratio(turns, duration_s)
        logger.info(f"Session {session_id} overlap ratio: {overlap_ratio:.2%}")
        
        # Step 4: Separating (Speech unmixing if overlap > 15%)
        await set_job_progress(job_id, 40, "separating")
        separator = get_separator()
        separated_tracks = [preprocessed_wav_path]
        
        if separator.should_separate(overlap_ratio):
            logger.info("Overlapping speech ratio exceeds 15%. Initiating source separation...")
            separated_tracks = separator.separate_speakers(preprocessed_wav_path)
            if len(separated_tracks) > 1:
                used_separation = True
                logger.info(f"Speech successfully unmixed into {len(separated_tracks)} tracks.")
                
        # Step 5: Transcribing
        await set_job_progress(job_id, 55, "transcribing")
        transcriber = get_transcriber()
        all_words = []
        
        if used_separation:
            # Transcribe separated tracks independently
            for spk_idx, track in enumerate(separated_tracks):
                speaker_label = f"SPEAKER_{spk_idx:02d}"
                track_chunks = chunk_audio(track)
                for chunk in track_chunks:
                    chunk_words = transcriber.transcribe_chunk(chunk["path"])
                    for cw in chunk_words:
                        cw["start"] += chunk["start_ms"] / 1000.0
                        cw["end"] += chunk["start_ms"] / 1000.0
                        cw["speaker"] = speaker_label
                        all_words.append(cw)
        else:
            # Transcribe original chunks
            chunks = chunk_audio(preprocessed_wav_path)
            for chunk in chunks:
                chunk_words = transcriber.transcribe_chunk(chunk["path"])
                for cw in chunk_words:
                    cw["start"] += chunk["start_ms"] / 1000.0
                    cw["end"] += chunk["start_ms"] / 1000.0
                    all_words.append(cw)
                    
        # Step 6: Aligning
        await set_job_progress(job_id, 75, "aligning")
        if used_separation:
            aligned_words = all_words
        else:
            aligned_words = align_words_to_speakers(all_words, turns)
            
        # Deduplicate overlapping word tokens
        deduped_words = AudioTranscriber.deduplicate_words(aligned_words)
        
        # Group into same-speaker utterances
        final_segments = group_into_segments(deduped_words)
        
        # Step 7: Saving to Database
        await set_job_progress(job_id, 85, "saving")
        async with AsyncSessionLocal() as db:
            session = await db.get(MeetingSession, uuid.UUID(session_id))
            if session:
                session.duration_s = duration_s
                session.status = "processing"
                
                # Proactively clean up any existing transcript for this session
                from sqlalchemy import delete
                await db.execute(delete(Transcript).where(Transcript.session_id == session.id))
                
                # Save new transcript
                transcript_record = Transcript(
                    session_id=session.id,
                    segments=final_segments
                )
                db.add(transcript_record)
                await db.commit()
                logger.info(f"Successfully saved transcript for session {session_id} to database.")
            else:
                logger.error(f"Meeting session {session_id} not found in database!")
                raise ValueError(f"Session {session_id} not found")
                
        # Step 8: Queue for LLM summaries
        await set_job_progress(job_id, 90, "queued_for_llm")
        await dispatch_llm_job(session_id)
        logger.info(f"Audio pipeline complete for session {session_id}. Dispatched LLM job.")
        
    except Exception as e:
        logger.error(f"Pipeline error encountered for session {session_id}: {e}", exc_info=True)
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
        
    finally:
        # Cleanup chunk files and preprocessed outputs to avoid disk bloating
        try:
            if preprocessed_wav_path:
                chunk_dir = preprocessed_wav_path + "_chunks"
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
                if os.path.exists(preprocessed_wav_path):
                    os.remove(preprocessed_wav_path)
            
            for track in separated_tracks:
                track_chunk_dir = track + "_chunks"
                if os.path.exists(track_chunk_dir):
                    shutil.rmtree(track_chunk_dir)
                if used_separation and track != preprocessed_wav_path and os.path.exists(track):
                    os.remove(track)
        except Exception as cleanup_err:
            logger.warning(f"Error cleaning up temporary audio files: {cleanup_err}")
