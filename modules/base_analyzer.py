"""
Common base class for all video analyzers.
Centralizes caption generation and validation logic.
Supports Gemini, Moondream, and NVIDIA Multimodal LLMs (NeMoVision, Llama Vision)
"""

import cv2
import numpy as np
from PIL import Image
import time
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import tempfile
import os
import requests

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import moondream as md
    MOONDREAM_AVAILABLE = True
except ImportError:
    MOONDREAM_AVAILABLE = False


class BaseVideoAnalyzer:
    """
    Base class for watcher.py and realtime.py.
    Contains all common video processing logic.
    Supports Gemini, Moondream, and NVIDIA Multimodal LLMs.
    """

    def __init__(self, config, mode: str = 'watcher'):
        """
        Initialize base analyzer.

        Args:
            config: Application configuration
            mode: 'watcher' or 'realtime'
        """
        self.config = config
        self.mode = mode

        self.gemini_client = None
        self.moondream_model = None
        self.nvidia_api_key = None
        self.nvidia_model = None

        self.current_model_type = None  # 'gemini', 'moondream', 'nemotron', 'llama90b', 'llama11b'
        self.temp_files = []
        self.api_error = False

        self.stats = {
            'frames_processed': 0,
            'captions_generated': 0,
            'captions_rejected': 0,
            'errors': 0,
            'last_error': None,
            'api_error_occurred': False
        }

    # ============================================================================
    # MODEL LOADING METHODS
    # ============================================================================

    def load_gemini_model(self, api_key: str = None) -> bool:
        """Load Gemini AI model (Multimodal LLM - captioning + QA)."""
        if not GEMINI_AVAILABLE:
            print("Google GenAI not available")
            return False

        try:
            api_key = api_key or self.config.GEMINI_API_KEY or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                print("No Gemini API key provided")
                return False

            self.gemini_client = genai.Client(api_key=api_key)
            self.current_model_type = 'gemini'
            self.api_error = False
            print("Gemini client initialized (Multimodal LLM - captioning + QA)")
            return True

        except Exception as e:
            print(f"Error loading Gemini: {e}")
            return False

    def load_moondream_model(self, api_key: str = None) -> bool:
        """Load Moondream model (VLM only - captioning only, no text QA)."""
        if not MOONDREAM_AVAILABLE:
            print("Moondream not available")
            return False

        try:
            api_key = api_key or self.config.MOONDREAM_API_KEY or os.environ.get('MOONDREAM_API_KEY')
            if not api_key:
                print("No Moondream API key provided")
                return False

            self.moondream_model = md.vl(api_key=api_key)
            self.current_model_type = 'moondream'
            self.api_error = False
            print("Moondream model initialized (VLM - captioning only)")
            return True

        except Exception as e:
            print(f"Error loading Moondream: {e}")
            return False

    def load_nvidia_model(self, model_name: str, api_key: str = None) -> bool:
        """
        Load NVIDIA Multimodal LLM (NeMoVision or Llama Vision).
        
        Args:
            model_name: One of:
                - 'nemotron' -> nvidia/nemotron-nano-12b-v2-vl
                - 'llama90b' -> meta/llama-3.2-90b-vision-instruct
                - 'llama11b' -> meta/llama-3.2-11b-vision-instruct
            api_key: NVIDIA API key (nvapi-...)
        
        Returns:
            True if successful
        """
        try:
            api_key = api_key or self.config.NVIDIA_API_KEY or os.environ.get('NVIDIA_API_KEY')
            if not api_key:
                print("No NVIDIA API key provided")
                return False

            self.nvidia_api_key = api_key
            
            if model_name == 'nemotron':
                self.nvidia_model = self.config.NEMOTRON_MODEL
                self.current_model_type = 'nemotron'
                print("NeMoVision loaded (Multimodal LLM - captioning + QA)")
            elif model_name == 'llama90b':
                self.nvidia_model = self.config.LLAMA_90B_VISION_MODEL
                self.current_model_type = 'llama90b'
                print("Llama 3.2 90B Vision loaded (Multimodal LLM - captioning + QA)")
            elif model_name == 'llama11b':
                self.nvidia_model = self.config.LLAMA_11B_VISION_MODEL
                self.current_model_type = 'llama11b'
                print("Llama 3.2 11B Vision loaded (Multimodal LLM - captioning + QA)")
            else:
                print(f"Unknown NVIDIA model: {model_name}")
                return False
            
            self.api_error = False
            return True

        except Exception as e:
            print(f"Error loading NVIDIA model: {e}")
            return False

    # ============================================================================
    # CAPTION GENERATION
    # ============================================================================

    def generate_caption(self, frame: np.ndarray, timestamp: float) -> Optional[str]:
        """Generate caption using current model."""
        if self.api_error:
            return None

        self.stats['frames_processed'] += 1

        if self.current_model_type == 'gemini':
            caption = self._generate_gemini_caption(frame)
        elif self.current_model_type == 'moondream':
            caption = self._generate_moondream_caption(frame)
        elif self.current_model_type in ['nemotron', 'llama90b', 'llama11b']:
            caption = self._generate_nvidia_caption(frame)
        else:
            print("No model loaded")
            return None

        if self.api_error:
            return None

        if caption and len(caption.strip()) > 0:
            self.stats['captions_generated'] += 1
            return caption.strip()
        else:
            self.stats['captions_rejected'] += 1
            return None

    def _generate_gemini_caption(self, frame: np.ndarray) -> Optional[str]:
        """Generate caption with Gemini (Multimodal LLM)."""
        if not self.gemini_client:
            return None

        temp_path = None
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_image.save(tmp.name, quality=85)
                temp_path = tmp.name
                self.temp_files.append(temp_path)

            with open(temp_path, 'rb') as f:
                image_bytes = f.read()

            response = self.gemini_client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    "You are a surveillance assistant. Describe this single video frame concisely in one sentence. "
                    "Focus on visible objects, count people (men, women), clothes colors, actions, and environment. "
                    "Be specific but brief."
                ]
            )
            return response.text.strip()

        except Exception as e:
            print(f"Gemini API error: {e}")
            self.api_error = True
            self.stats['api_error_occurred'] = True
            return None

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    if temp_path in self.temp_files:
                        self.temp_files.remove(temp_path)
                except:
                    pass

    def _generate_moondream_caption(self, frame: np.ndarray) -> Optional[str]:
        """Generate caption with Moondream (VLM only)."""
        if not self.moondream_model:
            return None

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            result = self.moondream_model.caption(pil_image)
            caption = result.get("caption", "")

            return caption.strip() if caption else None

        except Exception as e:
            print(f"Moondream API error: {e}")
            self.api_error = True
            self.stats['api_error_occurred'] = True
            return None

    def _generate_nvidia_caption(self, frame: np.ndarray) -> Optional[str]:
        """
        Generate caption with NVIDIA Multimodal LLM (NeMoVision or Llama Vision).
        Uses the same API endpoint as the reference code.
        """
        if not self.nvidia_api_key or not self.nvidia_model:
            return None

        temp_path = None
        try:
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_b64 = base64.b64encode(buffer).decode()

            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Content-Type": "application/json"
            }

            prompt = (
                "You are a surveillance assistant. Describe this single video frame concisely in one sentence. "
                "Focus on visible objects, count people (men, women), clothes colors, actions, and environment. "
                "Be specific but brief."
            )

            payload = {
                "model": self.nvidia_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3,
                "top_p": 0.95
            }

            response = requests.post(
                self.config.NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                print(f"NVIDIA API error: HTTP {response.status_code}")
                self.api_error = True
                self.stats['api_error_occurred'] = True
                return None

            result = response.json()
            caption = result["choices"][0]["message"]["content"]
            return caption.strip()

        except Exception as e:
            print(f"NVIDIA API error: {e}")
            self.api_error = True
            self.stats['api_error_occurred'] = True
            return None

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    # ============================================================================
    # Q&A METHODS (for Multimodal LLMs - Gemini and NVIDIA models)
    # ============================================================================

    def supports_qa(self) -> bool:
        """
        Check if current model supports text-based Q&A.
        
        Returns:
            True for Gemini and NVIDIA models (multimodal LLMs)
            False for Moondream (VLM only)
        """
        return self.current_model_type in ['gemini', 'nemotron', 'llama90b', 'llama11b']

    def answer_question(self, question: str, context_data: Dict) -> Dict:
        """
        Answer a question based on retrieved video segments.
        Works with Gemini and NVIDIA Multimodal LLMs.
        
        Args:
            question: User's question
            context_data: Dictionary with 'context' and 'results' from VectorMemory
        
        Returns:
            Dictionary with 'answer', 'error', etc.
        """
        if not self.supports_qa():
            return {
                'answer': "Q&A is only available with Multimodal LLMs (Gemini, NeMoVision, or Llama Vision). Please switch to one of these models for Q&A.",
                'error': True,
                'needs_multimodal_llm': True
            }

        if self.current_model_type == 'gemini':
            return self._answer_question_gemini(question, context_data)
        elif self.current_model_type in ['nemotron', 'llama90b', 'llama11b']:
            return self._answer_question_nvidia(question, context_data)
        else:
            return {
                'answer': f"Q&A not supported for {self.current_model_type}",
                'error': True
            }

    def _answer_question_gemini(self, question: str, context_data: Dict) -> Dict:
        """Answer question using Gemini."""
        if not self.gemini_client:
            return {'answer': 'Gemini model not loaded', 'error': True}

        try:
            prompt = self._build_qa_prompt(question, context_data)

            response = self.gemini_client.models.generate_content(
                model=self.config.GEMINI_MODEL,
                contents=prompt
            )

            answer = response.text.strip()

            return {
                'answer': answer,
                'model': 'Gemini 3',
                'segments_analyzed': len(context_data.get('results', [])),
                'error': False
            }

        except Exception as e:
            print(f"Gemini Q&A error: {str(e)}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'error': True
            }

    def _answer_question_nvidia(self, question: str, context_data: Dict) -> Dict:
        """Answer question using NVIDIA Multimodal LLM (NeMoVision or Llama Vision)."""
        if not self.nvidia_api_key or not self.nvidia_model:
            return {'answer': 'NVIDIA model not loaded', 'error': True}

        try:
            prompt = self._build_qa_prompt(question, context_data)

            # Get model display name
            model_display = {
                'nemotron': 'NeMoVision 12B',
                'llama90b': 'Llama 3.2 90B Vision',
                'llama11b': 'Llama 3.2 11B Vision'
            }.get(self.current_model_type, 'NVIDIA Vision Model')

            headers = {
                "Authorization": f"Bearer {self.nvidia_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.nvidia_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.95
            }

            response = requests.post(
                self.config.NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                return {
                    'answer': f"NVIDIA API error: HTTP {response.status_code}",
                    'error': True
                }

            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()

            return {
                'answer': answer,
                'model': model_display,
                'segments_analyzed': len(context_data.get('results', [])),
                'error': False
            }

        except Exception as e:
            print(f"NVIDIA Q&A error: {str(e)}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'error': True
            }

    def _build_qa_prompt(self, question: str, context_data: Dict) -> str:
        """Build the Q&A prompt with context."""
        context = context_data.get('context', 'No context available.')
        result_count = context_data.get('result_count', 0)

        prompt = f"""You are Vision-Talk, an intelligent video analysis assistant. You help users understand what's happening in their videos based on captions extracted from video frames.

CONTEXT FROM VIDEO DATABASE:
I have found {result_count} video segments that are semantically relevant to the question. Here are the captions from those segments with timestamps and relevance scores:

{context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Answer the question based ONLY on the video segments provided above
2. The segments are ordered by relevance (highest relevance first)
3. If the answer isn't in the segments, say you couldn't find relevant footage
4. Reference specific timestamps when you mention events (use HH:MM:SS format)
5. When referencing a segment, mention its number like [Segment 1], [Segment 2], etc.
6. When mentioning a video name, just use the filename as is
7. Be concise but thorough
8. If you notice patterns across multiple segments, mention them
9. Provide insights and observations based on the video content

ANSWER:
"""
        return prompt

    # ============================================================================
    # TIMESTAMP HANDLING
    # ============================================================================

    def format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def create_caption_data(self, caption: str, timestamp: float,
                            frame_count: int, video_source: str,
                            video_path: str = None) -> Dict[str, Any]:
        """Create standardized caption data dictionary."""
        data = {
            'caption': caption,
            'frame_count': frame_count,
            'video_source': video_source,
            'video_path': video_path or '',
            'timestamp_type': 'absolute' if self.mode == 'realtime' else 'elapsed',
            'timestamp_value': float(timestamp)
        }
        
        if self.mode == 'realtime':
            dt = datetime.fromtimestamp(timestamp)
            data['timestamp_display'] = dt.strftime("%H:%M:%S")
        else:
            data['timestamp_display'] = self.format_timestamp(timestamp)

        return data

    # ============================================================================
    # UTILITIES
    # ============================================================================

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information and metadata."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {}

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            return {
                'fps': float(fps),
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': float(frame_count / fps) if fps > 0 else 0,
                'file_size': Path(video_path).stat().st_size if Path(video_path).exists() else 0,
                'filename': Path(video_path).name,
                'file_path': str(video_path)
            }

        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return {}
        finally:
            if cap:
                cap.release()

    def extract_frame(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract a single frame from video at specified timestamp."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()

            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None

        except Exception as e:
            print(f"Error extracting frame: {e}")
            return None
        finally:
            if cap:
                cap.release()

    def cleanup_temp_files(self):
        """Clean up any temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")
        self.temp_files = []

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return self.stats.copy()
    
    def has_api_error(self) -> bool:
        """Check if API error occurred."""
        return self.api_error