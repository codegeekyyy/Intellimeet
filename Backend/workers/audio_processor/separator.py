import os
import logging
import wave
import numpy as np
from pydub import AudioSegment
from app.config import settings

logger = logging.getLogger("audio_worker.separator")

class AudioSeparator:
    def __init__(self):
        self.enabled = False
        self.model = None
        
        try:
            # Try importing speechbrain and torch
            from speechbrain.inference.separation import SepformerSeparation
            import torch
            
            save_dir = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "sepformer")
            os.makedirs(save_dir, exist_ok=True)
            
            logger.info("Loading SpeechBrain SepFormer separation model...")
            self.device = "cuda" if settings.USE_GPU and torch.cuda.is_available() else "cpu"
            
            self.model = SepformerSeparation.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=save_dir,
                run_opts={"device": self.device}
            )
            self.enabled = True
            logger.info("SpeechBrain SepFormer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SpeechBrain SepFormer model: {e}. "
                           f"Source separation will be bypassed (falling back to original audio).")
            self.model = None

    def should_separate(self, overlap_ratio: float) -> bool:
        """
        Check if source separation should run based on the detected speech overlap.
        """
        return self.enabled and overlap_ratio > 0.15

    def separate_speakers(self, wav_path: str) -> list[str]:
        """
        Separate mixed audio into individual speaker streams.
        Returns a list of file paths to the separated audio tracks.
        """
        if not self.enabled or self.model is None:
            logger.info("Source separation is disabled or model failed to load. Bypassing.")
            return [wav_path]
            
        logger.info(f"Running source separation on {wav_path}...")
        try:
            # Separate file - returns tensor of shape [batch, time, source]
            est_sources = self.model.separate_file(path=wav_path)
            
            # est_sources shape is [1, time_samples, num_sources]
            signal_tensor = est_sources[0] # shape [time_samples, num_sources]
            num_sources = signal_tensor.shape[1]
            
            out_paths = []
            base_dir = os.path.dirname(wav_path)
            filename = os.path.basename(wav_path)
            
            for i in range(num_sources):
                # Extract 1D numpy array for the source track
                source_signal = signal_tensor[:, i].cpu().numpy()
                
                # Normalize signal to prevent clipping
                max_val = np.max(np.abs(source_signal))
                if max_val > 0:
                    source_signal = source_signal / max_val
                    
                sep_filename = f"sep_source_{i}_{filename}"
                sep_path = os.path.join(base_dir, sep_filename)
                
                # SepFormer WSJ02mix outputs at 8000 Hz
                temp_8k_path = sep_path + ".8k.wav"
                self._save_numpy_to_wav(temp_8k_path, source_signal, sample_rate=8000)
                
                # Load with pydub and upsample to 16kHz for Whisper
                sound = AudioSegment.from_wav(temp_8k_path)
                sound = sound.set_frame_rate(16000)
                sound.export(sep_path, format="wav")
                
                # Clean up temporary 8k file
                if os.path.exists(temp_8k_path):
                    os.remove(temp_8k_path)
                    
                out_paths.append(sep_path)
                
            logger.info(f"Source separation complete. Generated: {out_paths}")
            return out_paths
            
        except Exception as e:
            logger.error(f"Error during source separation: {e}. Falling back to original audio.")
            return [wav_path]

    @staticmethod
    def _save_numpy_to_wav(filename: str, signal: np.ndarray, sample_rate: int = 8000):
        """
        Helper method to save a 1D float numpy array to a 16-bit PCM WAV file.
        """
        # Convert float signal [-1.0, 1.0] to 16-bit integer PCM
        pcm_signal = (signal * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2) # 2 bytes = 16 bits
            w.setframerate(sample_rate)
            w.writeframes(pcm_signal.tobytes())
