"""
Video analysis module for uploaded files.
Inherits from BaseVideoAnalyzer for common logic.
Supports Gemini, Moondream, and NVIDIA Multimodal LLMs.
"""

import cv2
import time
from typing import List, Dict, Optional, Callable
from pathlib import Path
import gc

from modules.base_analyzer import BaseVideoAnalyzer


class VideoWatcher(BaseVideoAnalyzer):
    """Analyzer for uploaded videos."""

    def __init__(self, config):
        """Initialize with configuration."""
        super().__init__(config, mode='watcher')

    # ============================================================================
    # MODEL LOADING (with NVIDIA support)
    # ============================================================================

    def load_model_by_name(self, model_name: str, api_key: str = None, nvidia_api_key: str = None) -> bool:
        """
        Load model by display name.
        
        Args:
            model_name: 'Gemini Flash 3', 'Moondream', 'NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision'
            api_key: API key for Gemini or Moondream
            nvidia_api_key: NVIDIA API key for NeMoVision/Llama models
        
        Returns:
            True if successful
        """
        if model_name == 'Gemini Flash 3':
            return self.load_gemini_model(api_key)
        elif model_name == 'Moondream':
            return self.load_moondream_model(api_key)
        elif model_name == 'NeMoVision':
            return self.load_nvidia_model('nemotron', nvidia_api_key)
        elif model_name == 'Llama 90B Vision':
            return self.load_nvidia_model('llama90b', nvidia_api_key)
        elif model_name == 'Llama 11B Vision':
            return self.load_nvidia_model('llama11b', nvidia_api_key)
        else:
            print(f"Unknown model: {model_name}")
            return False

    # ============================================================================
    # VIDEO PROCESSING
    # ============================================================================

    def process_video(self, video_path: str, frame_interval: int = None,
                      max_frames: int = None,
                      on_caption_generated: Callable = None) -> List[Dict]:
        """Process video and generate captions for frames at specified intervals."""
        if frame_interval is None:
            frame_interval = self.config.FRAME_INTERVAL

        results = []
        cap = None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_skip = max(1, int(fps * frame_interval))

            # Model display name
            model_display = {
                'gemini': 'Gemini',
                'moondream': 'Moondream',
                'nemotron': 'NeMoVision',
                'llama90b': 'Llama 90B Vision',
                'llama11b': 'Llama 11B Vision'
            }.get(self.current_model_type, self.current_model_type or 'Unknown')

            print(f"\n🎬 Processing: {Path(video_path).name}")
            print(f"📊 FPS: {fps:.2f}, Total frames: {total_frames}")
            print(f"⏱️  Interval: {frame_interval}s (skip {frames_to_skip} frames)")
            print(f"🤖 Model: {model_display} ({'Multimodal LLM' if self.supports_qa() else 'VLM'})\n")

            results = self._process_frames(
                cap, video_path, fps, frames_to_skip,
                max_frames, on_caption_generated
            )

            if self.api_error:
                print(f"\n⚠️ Processing stopped due to API error")
            else:
                print(f"\n✅ Processing complete: {len(results)} captions generated")
            
            return results

        except Exception as e:
            print(f"❌ Processing error: {e}")
            return []

        finally:
            if cap:
                cap.release()
            self.cleanup_temp_files()
            gc.collect()

    def _process_frames(self, cap, video_path, fps, frames_to_skip,
                        max_frames=None, on_caption_generated=None):
        """Internal frame processing."""
        results = []
        frame_count = 0
        processed_count = 0
        start_time = time.time()

        while True:
            if self.api_error:
                print(f"\n⚠️ API error occurred - stopping further processing")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frames_to_skip == 0:
                processed_count += 1

                if max_frames and processed_count > max_frames:
                    break

                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"📊 Processed: {processed_count} frames | Time: {elapsed:.1f}s")

                current_time = frame_count / fps
                caption = self.generate_caption(frame, current_time)

                if caption:
                    result = self.create_caption_data(
                        caption=caption,
                        timestamp=current_time,
                        frame_count=frame_count,
                        video_source=Path(video_path).name,
                        video_path=video_path
                    )

                    results.append(result)

                    if on_caption_generated:
                        try:
                            on_caption_generated(result)
                        except Exception as e:
                            print(f"⚠️ Callback error: {e}")

                    timestamp_str = self.format_timestamp(current_time)
                    print(f"  ✅ [{timestamp_str}] {caption[:60]}...")

            frame_count += 1

        return results

    # ============================================================================
    # Q&A METHOD 
    # ============================================================================

    def answer_question(self, question: str, context_data: Dict) -> Dict:
        """
        Answer a question using the loaded multimodal LLM.
        Delegates to base class implementation.
        """
        return super().answer_question(question, context_data)