import os
import uuid
import shutil
from app.config import settings

async def save_audio_locally(tmp_path: str, user_id: str) -> str:
    """
    Moves a temporary uploaded audio file to the permanent upload folder.
    Returns the relative path to store in the database.
    """
    ext = os.path.splitext(tmp_path)[1]
    filename = f"{user_id}/{uuid.uuid4()}{ext}"
    dest = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Ensure user subfolder exists
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Move temp file to permanent destination
    shutil.move(tmp_path, dest)
    return filename

def get_audio_full_path(relative_path: str) -> str:
    """Resolve a relative database path to an absolute path on disk."""
    return os.path.abspath(os.path.join(settings.UPLOAD_DIR, relative_path))

def delete_audio_file(relative_path: str) -> None:
    """Permanently delete an audio file from local storage."""
    try:
        full_path = get_audio_full_path(relative_path)
        if os.path.exists(full_path):
            os.remove(full_path)
    except Exception as e:
        import logging
        logging.getLogger("intellimeet").error(f"Failed to delete audio file: {e}")
