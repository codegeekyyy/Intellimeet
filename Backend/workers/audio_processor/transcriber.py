import os
import logging
from faster_whisper import WhisperModel
from app.config import settings

# Add WinGet FFmpeg path to PATH dynamically in case faster-whisper invokes ffmpeg/ffprobe subprocesses
ffmpeg_bin_path = r"C:\Users\harsh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
if os.path.exists(ffmpeg_bin_path) and ffmpeg_bin_path not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ["PATH"]

# Add Python-packaged NVIDIA DLL directories to PATH for ctranslate2 (faster-whisper) on Windows
import sys
if sys.platform == "win32":
    venv_path = sys.prefix
    site_packages = os.path.join(venv_path, "Lib", "site-packages")
    if os.path.exists(site_packages):
        for pkg in ["cublas", "cudnn", "cuda_runtime", "cusolver", "cusparse"]:
            bin_dir = os.path.join(site_packages, "nvidia", pkg, "bin")
            if os.path.exists(bin_dir) and bin_dir not in os.environ["PATH"]:
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]


logger = logging.getLogger("audio_worker.transcriber")

class AudioTranscriber:
    def __init__(self, model_size: str = None, use_gpu: bool = None):
        # Allow passing config parameters, but default to settings
        model_size = model_size or settings.WHISPER_MODEL
        use_gpu = settings.USE_GPU if use_gpu is None else use_gpu
        
        self.device = "cuda" if use_gpu else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"Initializing Whisper model '{model_size}' on device '{self.device}' (compute_type: {self.compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        logger.info("Whisper model initialized successfully.")

    def transcribe_chunk(self, wav_path: str, offset_ms: int = 0, language: str = None) -> list[dict]:
        """
        Transcribe an audio chunk and extract word-level timestamps,
        offsetting them by the chunk's start time (in milliseconds).
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Chunk audio file not found: {wav_path}")
            
        segments, info = self.model.transcribe(
            wav_path,
            beam_size=5,
            word_timestamps=True,
            language=language,
            vad_filter=True
        )
        
        words_list = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words_list.append({
                        "word": word.word.strip(),
                        "start": word.start + (offset_ms / 1000.0),
                        "end": word.end + (offset_ms / 1000.0),
                        "probability": word.probability
                    })
        return words_list

    @staticmethod
    def deduplicate_words(words: list[dict], match_window_sec: float = 0.5) -> list[dict]:
        """
        Deduplicate transcribed words from adjacent overlapping chunks.
        Expects words to have absolute 'start' and 'end' time values.
        """
        if not words:
            return []
            
        # Sort words chronologically by their absolute start times
        sorted_words = sorted(words, key=lambda w: w["start"])
        deduped = [sorted_words[0]]
        
        for w in sorted_words[1:]:
            is_dup = False
            w_clean = w["word"].strip().lower().strip(".,?!:;\"'-")
            
            # Check backwards in time
            for prev in reversed(deduped):
                # If we've moved past the comparison time window, stop searching
                if prev["end"] < w["start"] - match_window_sec:
                    break
                    
                prev_clean = prev["word"].strip().lower().strip(".,?!:;\"'-")
                # If start times are within the window and the word content matches
                if abs(prev["start"] - w["start"]) < match_window_sec and prev_clean == w_clean:
                    is_dup = True
                    # Keep the one with higher confidence probability
                    if w.get("probability", 0) > prev.get("probability", 0):
                        prev["start"] = w["start"]
                        prev["end"] = w["end"]
                        prev["probability"] = w["probability"]
                        prev["word"] = w["word"]
                    break
                    
            if not is_dup:
                deduped.append(w)
                
        return deduped