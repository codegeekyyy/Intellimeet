import os
import logging
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from app.config import settings

logger = logging.getLogger("audio_worker.diarizer")

class AudioDiarizer:
    def __init__(self):
        self.device = torch.device("cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu")
        logger.info(f"Loading pyannote speaker diarization model on {self.device}...")
        
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=settings.HUGGINGFACE_TOKEN
            )
            if self.pipeline:
                self.pipeline.to(self.device)
                logger.info("Diarization pipeline loaded successfully.")
            else:
                logger.warning("Pipeline returned None. Falling back to single-speaker mode.")
                self.pipeline = None
        except Exception as e:
            logger.warning(f"Could not load pyannote diarization pipeline: {e}. "
                           f"Make sure HUGGINGFACE_TOKEN is set in .env and you have accepted the "
                           f"pyannote/speaker-diarization-3.1 user agreement on Hugging Face. "
                           f"Falling back to single-speaker mode.")
            self.pipeline = None

    def diarize(self, wav_path: str, num_speakers: int = None) -> list[dict]:
        """
        Run speaker diarization on a WAV file.
        Returns a list of speaker turns: [{"speaker": str, "start": float, "end": float}]
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"WAV file not found for diarization: {wav_path}")
            
        if self.pipeline is None:
            # Fallback to single speaker spanning the duration of the audio
            audio = AudioSegment.from_wav(wav_path)
            duration_s = len(audio) / 1000.0
            logger.info(f"Fallback: Single speaker (SPEAKER_00) diarization for {duration_s}s.")
            return [{"speaker": "SPEAKER_00", "start": 0.0, "end": duration_s}]

        logger.info(f"Running diarization on {wav_path}...")
        
        # Run pyannote pipeline
        diarization = self.pipeline(wav_path, num_speakers=num_speakers)
        
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
            
        logger.info(f"Diarization complete. Found {len(turns)} speaker turns.")
        return turns

def align_words_to_speakers(words: list[dict], turns: list[dict]) -> list[dict]:
    """
    Map each word to a speaker label based on the speaker turns.
    Uses the midpoint of the word to determine which speaker turn it falls into.
    """
    aligned_words = []
    
    for w in words:
        midpoint = (w["start"] + w["end"]) / 2.0
        assigned_speaker = "UNKNOWN"
        
        # Check if midpoint falls inside any turn
        for turn in turns:
            if turn["start"] <= midpoint <= turn["end"]:
                assigned_speaker = turn["speaker"]
                break
                
        # If no turn matches exactly, find the closest turn
        if assigned_speaker == "UNKNOWN" and turns:
            closest_turn = min(
                turns, 
                key=lambda t: min(abs(midpoint - t["start"]), abs(midpoint - t["end"]))
            )
            assigned_speaker = closest_turn["speaker"]
            
        aligned_words.append({
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "probability": w["probability"],
            "speaker": assigned_speaker
        })
        
    return aligned_words

def group_into_segments(aligned_words: list[dict], max_gap_sec: float = 1.5) -> list[dict]:
    """
    Group consecutive words from the same speaker into utterance segments.
    Splits the segment if:
      - The speaker changes.
      - The silence gap between words exceeds max_gap_sec.
    """
    if not aligned_words:
        return []
        
    segments = []
    current_words = [aligned_words[0]]
    current_speaker = aligned_words[0]["speaker"]
    current_start = aligned_words[0]["start"]
    
    for w in aligned_words[1:]:
        gap = w["start"] - current_words[-1]["end"]
        
        if w["speaker"] == current_speaker and gap <= max_gap_sec:
            current_words.append(w)
        else:
            # Save the current group of words as a segment
            text = "".join([cw["word"] for cw in current_words]).strip()
            segments.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": current_words[-1]["end"],
                "text": text,
                "words": current_words
            })
            
            # Start a new segment group
            current_words = [w]
            current_speaker = w["speaker"]
            current_start = w["start"]
            
    # Save the final segment
    if current_words:
        text = "".join([cw["word"] for cw in current_words]).strip()
        segments.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_words[-1]["end"],
            "text": text,
            "words": current_words
        })
        
    return segments
