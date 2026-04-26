"""
Configuration module for Vision-Talk application.
Defines paths, model configurations, and processing parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    
    # ============================================================================
    # PATH CONFIGURATIONS
    # ============================================================================
    
    BASE_DIR = Path(__file__).parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    DOWNLOAD_FOLDER = BASE_DIR / "downloads"
    CHROMA_DB_PATH = BASE_DIR / "chroma_db"
    
    # ============================================================================
    # MODEL CONFIGURATIONS
    # ============================================================================
    
    EMBEDDING_MODEL = r"C:\Users\Wassim\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    RERANKING_MODEL = r"C:\Users\Wassim\.cache\huggingface\hub\models--cross-encoder--ms-marco-MiniLM-L-6-v2\snapshots\c5ee24cb16019beea0893ab7796b1df96625c6b8"
    GEMINI_MODEL = "models/gemini-3-flash-preview"
    MOONDREAM_MODEL = "Moondream-3-Preview"
    
    # NVIDIA API Models (Multimodal LLMs - both captioning AND QA)
    NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    NEMOTRON_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
    LLAMA_90B_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
    LLAMA_11B_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
    
    # ============================================================================
    # PROCESSING PARAMETERS
    # ============================================================================
    
    FRAME_INTERVAL = 5
    COLLECTION_NAME = "vision_talk_videos"
    
    QA_MAX_SEGMENTS = 15
    RERANKING_MULTIPLIER = 3
    REALTIME_MAX_QUEUE_SIZE = 5
    REALTIME_MAX_BUFFER_SIZE = 3
    
    # ============================================================================
    # API CONFIGURATION
    # ============================================================================
    
    GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
    MOONDREAM_API_KEY = os.environ.get('MOONDREAM_API_KEY', '')
    NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', '')
    
    def __init__(self):
        """Initialize configuration and create necessary directories."""
        self._create_directories()
        self._clean_download_folder()
    
    def _create_directories(self):
        """Create all necessary directories."""
        self.UPLOAD_FOLDER.mkdir(exist_ok=True)
        self.DOWNLOAD_FOLDER.mkdir(exist_ok=True)
        self.CHROMA_DB_PATH.mkdir(exist_ok=True)
    
    def _clean_download_folder(self):
        """Remove existing files from download folder on startup."""
        try:
            for file in self.DOWNLOAD_FOLDER.glob("*"):
                if file.is_file():
                    file.unlink()
        except Exception as e:
            print(f"⚠️ Could not clean download folder: {e}")