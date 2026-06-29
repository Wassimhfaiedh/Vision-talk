"""
Configuration module for Vision-Talk application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    DOWNLOAD_FOLDER = BASE_DIR / "downloads"
    RECORDINGS_FOLDER = BASE_DIR / "recordings"
    CHROMA_DB_PATH = BASE_DIR / "chroma_db"
    
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL_PATH', '')
    RERANKING_MODEL = os.environ.get('RERANKING_MODEL_PATH', '')
    GEMINI_MODEL = "models/gemini-3-flash-preview"
    MOONDREAM_MODEL = "Moondream-3-Preview"
    
    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
    LLAMA_90B_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
    LLAMA_11B_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
    
    FRAME_INTERVAL = 5
    COLLECTION_NAME = "vision_talk_videos"
    MAX_CONTENT_LENGTH = None
    
    QA_MAX_SEGMENTS = 15
    RERANKING_MULTIPLIER = 3
    REALTIME_MAX_QUEUE_SIZE = 5
    REALTIME_MAX_BUFFER_SIZE = 3
    
    GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
    MOONDREAM_API_KEY = os.environ.get('MOONDREAM_API_KEY', '')
    NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
    
    SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
    SMTP_EMAIL = os.environ.get('SMTP_EMAIL', '')
    SMTP_PASSWORD = os.environ.get('SMTP_PASSWORD', '')
    
    def __init__(self):
        self._create_directories()
        self._clean_download_folder()
    
    def _create_directories(self):
        self.UPLOAD_FOLDER.mkdir(exist_ok=True)
        self.DOWNLOAD_FOLDER.mkdir(exist_ok=True)
        self.RECORDINGS_FOLDER.mkdir(exist_ok=True)
        self.CHROMA_DB_PATH.mkdir(exist_ok=True)
    
    def _clean_download_folder(self):
        try:
            for file in self.DOWNLOAD_FOLDER.glob("*"):
                if file.is_file():
                    file.unlink()
        except Exception as e:
            print(f"Could not clean download folder: {e}")