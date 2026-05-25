import os
import sys
import asyncio
import logging
import shutil
from pydub.generators import Sine

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_audio_pipeline")

async def main():
    logger.info("=== STARTING AUDIO PIPELINE STEP VERIFICATION ===")
    
    # 1. Generate a small 5-second sine wave WAV file for testing
    test_file = "test_input.wav"
    logger.info(f"Generating 5-second test tone -> {test_file}...")
    tone = Sine(440).to_audio_segment(duration=5000)
    tone.export(test_file, format="wav")
    
    preprocessed_path = None
    
    try:
        # Step 1: Preprocessor test
        logger.info("\n--- STEP 1: Preprocessor Test ---")
        from workers.audio_processor.preprocessor import preprocess_audio, chunk_audio
        preprocessed_path = preprocess_audio(test_file)
        logger.info(f"Successfully preprocessed: {preprocessed_path}")
        
        chunks = chunk_audio(preprocessed_path)
        logger.info(f"Chunked WAV into {len(chunks)} overlapping parts.")
        for idx, chunk in enumerate(chunks):
            logger.info(f"  Chunk {idx}: {chunk}")
            
        # Step 3: Diarizer test
        logger.info("\n--- STEP 3: Diarizer Test ---")
        from workers.audio_processor.diarizer import AudioDiarizer, align_words_to_speakers, group_into_segments
        diarizer = AudioDiarizer()
        turns = diarizer.diarize(preprocessed_path)
        logger.info(f"Speaker turns: {turns}")
        
        # Step 4: Separator test
        logger.info("\n--- STEP 4: Separator Test ---")
        from workers.audio_processor.separator import AudioSeparator
        separator = AudioSeparator()
        should_sep = separator.should_separate(0.20) # test with 20% mock overlap
        logger.info(f"Should run separation for 20% overlap: {should_sep}")
        
        # Step 2: Transcriber test
        logger.info("\n--- STEP 2: Transcriber Test ---")
        from workers.audio_processor.transcriber import AudioTranscriber
        transcriber = AudioTranscriber()
        
        if chunks:
            chunk_path = chunks[0]["path"]
            logger.info(f"Transcribing first chunk: {chunk_path}...")
            words = transcriber.transcribe_chunk(chunk_path)
            logger.info(f"Transcribed words: {words}")
            
            # Process timestamps
            for w in words:
                w["start"] += chunks[0]["start_ms"] / 1000.0
                w["end"] += chunks[0]["start_ms"] / 1000.0
                
            # Align words to speaker labels
            aligned = align_words_to_speakers(words, turns)
            logger.info(f"Aligned speaker words: {aligned}")
            
            # Deduplicate words
            deduped = AudioTranscriber.deduplicate_words(aligned)
            logger.info(f"Deduplicated words: {deduped}")
            
            # Group into segments
            segments = group_into_segments(deduped)
            logger.info(f"Grouped segments: {segments}")
            
        logger.info("\n=== ALL AUDIO PIPELINE STEPS SUCCESSFULLY VERIFIED ===")
        
    except Exception as e:
        logger.error(f"Error occurred during step verification: {e}", exc_info=True)
        
    finally:
        # Cleanup files
        logger.info("Cleaning up temporary test files...")
        if os.path.exists(test_file):
            os.remove(test_file)
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        if preprocessed_path and os.path.exists(preprocessed_path + "_chunks"):
            shutil.rmtree(preprocessed_path + "_chunks")

if __name__ == "__main__":
    asyncio.run(main())
