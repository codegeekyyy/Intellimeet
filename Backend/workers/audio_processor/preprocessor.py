import os
import sys
import logging
import ffmpeg
from pydub import AudioSegment

# Add WinGet FFmpeg path to PATH dynamically so it doesn't require a system restart
ffmpeg_bin_path = r"C:\Users\harsh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin"
if os.path.exists(ffmpeg_bin_path) and ffmpeg_bin_path not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ["PATH"]

logger = logging.getLogger("audio_worker.preprocessor")

def preprocess_audio(input_path: str) -> str:
    """
    Normalize audio volume and convert to 16kHz mono WAV format via FFmpeg.
    Returns the path to the preprocessed WAV file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
    out_path = input_path + ".preprocessed.wav"
    logger.info(f"Preprocessing {input_path} -> {out_path}...")
    
    try:
        # Use ffmpeg-python to run conversion
        # ar=16000 (16kHz), ac=1 (mono), loudnorm (normalization), pcm_s16le codec (standard WAV)
        ffmpeg.input(input_path).output(
            out_path,
            ar=16000,
            ac=1,
            af="loudnorm",
            acodec="pcm_s16le"
        ).overwrite_output().run(quiet=True)
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error preprocessing audio: {e.stderr.decode() if e.stderr else str(e)}")
        raise RuntimeError(f"FFmpeg conversion failed: {e}")
        
    return out_path

def chunk_audio(wav_path: str) -> list[dict]:
    """
    Slice a WAV file into 30-second chunks with a 4-second overlap.
    Returns a list of dicts: [{"path": str, "start_ms": int, "end_ms": int}]
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file for chunking not found: {wav_path}")
        
    logger.info(f"Chunking {wav_path} into 30s windows with 4s overlap...")
    
    audio = AudioSegment.from_wav(wav_path)
    duration_ms = len(audio)
    
    chunk_length_ms = 30000  # 30 seconds
    overlap_ms = 4000        # 4 seconds
    step_ms = chunk_length_ms - overlap_ms  # 26 seconds
    
    chunks = []
    chunk_dir = wav_path + "_chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    
    start_ms = 0
    chunk_idx = 0
    
    while start_ms < duration_ms:
        end_ms = min(start_ms + chunk_length_ms, duration_ms)
        
        chunk_segment = audio[start_ms:end_ms]
        chunk_name = f"chunk_{chunk_idx:04d}_{start_ms}_{end_ms}.wav"
        chunk_path = os.path.join(chunk_dir, chunk_name)
        
        chunk_segment.export(chunk_path, format="wav")
        
        chunks.append({
            "path": chunk_path,
            "start_ms": start_ms,
            "end_ms": end_ms
        })
        
        if end_ms == duration_ms:
            break
            
        start_ms += step_ms
        chunk_idx += 1
        
    logger.info(f"Successfully generated {len(chunks)} chunks in {chunk_dir}.")
    return chunks
