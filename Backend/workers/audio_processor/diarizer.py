import os
import logging
import torch

# MONKEY PATCH: pyannote.audio.core.io uses torchcodec which is missing or fails to build/load on Windows.
# Since torchaudio's load function also tries to invoke torchcodec internally when it's present,
# we patch pyannote.audio.core.io using soundfile (which is robust and doesn't depend on torchcodec/torchaudio.load).
try:
    import pyannote.audio.core.io as pyannote_io
    import soundfile
    import torchaudio
    import torch.nn.functional as F
    from types import SimpleNamespace
    from io import IOBase

    def patched_get_audio_metadata(file):
        path = file["audio"]
        info = soundfile.info(path)
        return SimpleNamespace(
            sample_rate=info.samplerate,
            duration_seconds_from_header=info.duration
        )

    def patched_audio_call(self, file):
        file = self.validate_file(file)
        channel = file.get("channel", None)

        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]
            return self.downmix_and_resample(waveform, sample_rate, channel=channel)

        data, sample_rate = soundfile.read(file["audio"], dtype='float32')
        waveform = torch.from_numpy(data)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if isinstance(file["audio"], IOBase):
            file["audio"].seek(0)

        return self.downmix_and_resample(waveform, sample_rate, channel=channel)

    def patched_audio_crop(self, file, segment, mode="raise"):
        file = self.validate_file(file)
        channel = file.get("channel", None)

        if "waveform" in file:
            waveform = file["waveform"]
            _, num_samples = waveform.shape
            sample_rate = file["sample_rate"]
            duration = num_samples / sample_rate

            start_sample: int = self.get_num_samples(segment.start, sample_rate)
            pad_start: int = max(0, -start_sample)
            if start_sample < 0:
                if mode == "raise":
                    raise ValueError(f"requested chunk with negative start time (t={segment.start:.3f}s)")
                else:
                    start_sample = 0

            end_sample: int = self.get_num_samples(segment.end, sample_rate)
            pad_end: int = max(end_sample, num_samples) - num_samples
            if end_sample >= num_samples:
                if mode == "raise":
                    raise ValueError(
                        f"requested chunk with end time (t={segment.end:.3f}s) greater than "
                        f"{file.get('uri', 'in-memory')} file duration ({duration:.3f}s)."
                    )
                else:
                    end_sample = num_samples

            data = waveform[:, start_sample:end_sample]
            data = F.pad(data, (pad_start, pad_end))
            return self.downmix_and_resample(data, sample_rate, channel=channel)

        # soundfile crop implementation
        info = soundfile.info(file["audio"])
        sample_rate = info.samplerate
        duration = info.duration
        num_samples = info.frames

        start: float = float(segment.start)
        end: float = float(segment.end)

        pad_start: int = max(0, self.get_num_samples(-start, sample_rate))
        if start < 0:
            if mode == "raise":
                raise ValueError(f"requested chunk with negative start time (t={start:.3f}s)")
            else:
                start = 0.0

        pad_end: int = max(self.get_num_samples(end, sample_rate), num_samples) - num_samples
        if end > duration:
            if mode == "raise":
                raise ValueError(
                    f"requested chunk with end time (t={end:.3f}s) greater than "
                    f"{file.get('uri', 'in-memory')} file duration ({duration:.3f}s)."
                )
            else:
                end = duration

        frame_offset = self.get_num_samples(start, sample_rate)
        num_frames_to_load = self.get_num_samples(end - start, sample_rate)

        if frame_offset < 0:
            frame_offset = 0
        if frame_offset + num_frames_to_load > num_samples:
            num_frames_to_load = num_samples - frame_offset

        data, sr = soundfile.read(
            file["audio"],
            start=frame_offset,
            frames=num_frames_to_load,
            dtype='float32'
        )
        
        waveform = torch.from_numpy(data)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

        if isinstance(file["audio"], IOBase):
            file["audio"].seek(0)

        expected_num_samples = self.get_num_samples(segment.duration, sample_rate)
        _, actual_num_samples = waveform.shape
        difference = pad_start + actual_num_samples + pad_end - expected_num_samples
        if abs(difference) > 1:
            raise ValueError(
                f"requested chunk {segment} from {file.get('uri', 'in-memory')} file resulted in {actual_num_samples} samples "
                f"instead of the expected {expected_num_samples} samples."
            )

        if difference == 1:
            waveform = waveform[:, :-1]
        elif difference == -1:
            pad_end += 1

        waveform = F.pad(waveform, (pad_start, pad_end))
        return self.downmix_and_resample(waveform, sample_rate, channel=channel)

    pyannote_io.get_audio_metadata = patched_get_audio_metadata
    pyannote_io.Audio.__call__ = patched_audio_call
    pyannote_io.Audio.crop = patched_audio_crop
    logging.getLogger("audio_worker.diarizer").info("Successfully monkey-patched pyannote.audio.core.io to use soundfile.")
except Exception as patch_err:
    logging.getLogger("audio_worker.diarizer").error(f"Failed to monkeypatch pyannote.audio.core.io: {patch_err}")

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
        
        # In newer pyannote versions, the pipeline returns a DiarizeOutput object.
        # We extract speaker_diarization if present, otherwise assume it's already an Annotation.
        annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization

        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
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
