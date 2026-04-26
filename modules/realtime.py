"""
Real-time video analysis module for webcam stream.
Inherits from BaseVideoAnalyzer with queue and threading management.
Immediate database storage.
Records webcam video and saves as H.264 mp4 on stop.
Supports Gemini, Moondream, and NVIDIA Multimodal LLMs.
"""

import cv2
import numpy as np
import threading
import queue
import time
import subprocess
import os
from datetime import datetime
from typing import Optional, List, Dict, Callable
from pathlib import Path

from modules.base_analyzer import BaseVideoAnalyzer


class CircularBuffer:
    """
    Fixed-size ring buffer with O(1) push and latest access.
    Automatically overwrites oldest element when full.
    Thread-safe implementation using explicit locking.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._buffer = [None] * capacity
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()

    def push(self, item: object) -> None:
        with self._lock:
            self._buffer[self._head] = item
            self._head = (self._head + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1

    def latest(self) -> Optional[object]:
        with self._lock:
            if self._size == 0:
                return None
            latest_idx = (self._head - 1) % self._capacity
            return self._buffer[latest_idx]

    def get_all(self) -> List[object]:
        with self._lock:
            if self._size == 0:
                return []
            result = []
            start_idx = (self._head - self._size) % self._capacity
            for offset in range(self._size):
                idx = (start_idx + offset) % self._capacity
                result.append(self._buffer[idx])
            return result

    def clear(self) -> None:
        with self._lock:
            for i in range(self._capacity):
                self._buffer[i] = None
            self._head = 0
            self._size = 0

    @property
    def size(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def is_empty(self) -> bool:
        with self._lock:
            return self._size == 0

    def is_full(self) -> bool:
        with self._lock:
            return self._size == self._capacity


class RealTimeAnalyzer(BaseVideoAnalyzer):
    """
    Analyzer for real-time webcam stream with video recording.
    """

    def __init__(self, config, model_type: str = 'moondream',
                 api_key: str = None, nvidia_api_key: str = None,
                 frame_interval: int = 5, memory=None):
        super().__init__(config, mode='realtime')

        self.model_type = model_type.lower()
        self.frame_interval = frame_interval
        self.memory = memory

        self.frame_queue = queue.Queue(maxsize=config.REALTIME_MAX_QUEUE_SIZE)
        self.caption_queue = queue.Queue()
        self.is_running = False
        self.analysis_thread = None

        self.current_caption = f"Initializing {model_type}..."
        self.last_analysis_time = 0.0

        self._frame_buffer = CircularBuffer(capacity=config.REALTIME_MAX_BUFFER_SIZE)

        self.captions_data = []
        self.processed_count = 0
        self.start_time = None

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"webcam_{self.session_id}"
        self.on_new_caption: Optional[Callable] = None

        # ── ffmpeg-pipe recording state ────────────────────────────────────────
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._ffmpeg_lock = threading.Lock()      
        self._saved_video_path: Optional[str] = None

        # Recording parameters
        self._record_fps: float = 15.0           
        self._frame_size = (640, 360)            

        self._frames_written = 0
        self._frames_written_lock = threading.Lock()

        # Throttle: only send a frame to ffmpeg every 1/fps seconds
        self._min_write_interval = 1.0 / self._record_fps
        self._last_write_time: float = 0.0

        self._load_model(api_key, nvidia_api_key)

    # ── model loading ──────────────────────────────────────────────────────────

    def _load_model(self, api_key=None, nvidia_api_key=None) -> bool:
        if self.model_type == 'gemini':
            return self.load_gemini_model(api_key)
        elif self.model_type == 'moondream':
            return self.load_moondream_model(api_key)
        elif self.model_type == 'nemotron':
            return self.load_nvidia_model('nemotron', nvidia_api_key)
        elif self.model_type == 'llama90b':
            return self.load_nvidia_model('llama90b', nvidia_api_key)
        elif self.model_type == 'llama11b':
            return self.load_nvidia_model('llama11b', nvidia_api_key)
        else:
            print(f"Unknown model type: {self.model_type}")
            return False

    # ── ffmpeg pipe helpers ────────────────────────────────────────────────────

    @staticmethod
    def _ffmpeg_available() -> bool:
        """Return True if ffmpeg is on PATH."""
        try:
            subprocess.run(['ffmpeg', '-version'],
                           capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def _open_ffmpeg_pipe(self, out_path: str) -> Optional[subprocess.Popen]:
        """
        Open an ffmpeg subprocess that reads raw BGR frames from stdin
        and encodes them directly to an H.264 MP4 at *out_path*.

        Input format
        ------------
        rawvideo, bgr24, size=WxH, rate=_record_fps
        ffmpeg converts bgr24 -> yuv420p internally before encoding.
        """
        w, h = self._frame_size
        cmd = [
            'ffmpeg', '-y',
            # ── input: raw BGR frames from stdin ──────────────────────────────
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{w}x{h}',
            '-r', str(int(self._record_fps)),
            '-i', 'pipe:0',
            # ── output: H.264 MP4 ─────────────────────────────────────────────
            '-vcodec', 'libx264',
            '-preset', 'veryfast',      
            '-crf', '23',
            '-pix_fmt', 'yuv420p',     
            '-movflags', '+faststart',  
            '-an',                      
            out_path
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  
            )
            print(f"🎞️  ffmpeg pipe opened -> {out_path}")
            return proc
        except FileNotFoundError:
            print("⚠️ ffmpeg not found – recording disabled")
            return None
        except Exception as e:
            print(f"⚠️ ffmpeg pipe error: {e}")
            return None

    def _close_ffmpeg_pipe(self):
        """
        Cleanly close the ffmpeg stdin pipe and wait for encoding to finish.
        Must be called from stop() after all frames have been written.
        """
        with self._ffmpeg_lock:
            proc = self._ffmpeg_proc
            self._ffmpeg_proc = None

        if proc is None:
            return

        try:
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass

        
        try:
            proc.wait(timeout=300)
            print("✅ ffmpeg encoding complete")
        except subprocess.TimeoutExpired:
            proc.kill()
            print("⚠️ ffmpeg timed out – killed")
        except Exception as e:
            print(f"⚠️ ffmpeg wait error: {e}")

    # ============================================================================
    # SESSION LIFECYCLE
    # ============================================================================

    def start(self) -> str:
        self.is_running = True
        self.start_time = time.time()
        self.captions_data = []
        self.processed_count = 0
        self.api_error = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"webcam_{self.session_id}"
        self._saved_video_path = None
        self._frames_written = 0
        self._last_write_time = 0.0
        self._frame_buffer.clear()

        # Destination MP4 path (written directly by ffmpeg – no AVI temp file)
        out_path = str(self.config.UPLOAD_FOLDER / f"{self.session_name}.mp4")
        self._saved_video_path = out_path   # set early so stop() always has it

        if self._ffmpeg_available():
            self._ffmpeg_proc = self._open_ffmpeg_pipe(out_path)
        else:
            print("⚠️ ffmpeg not available – recording disabled")
            self._ffmpeg_proc = None

        # Start background analysis thread
        self.analysis_thread = threading.Thread(target=self._process_frames)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

        model_display = {
            'gemini': 'Gemini',
            'moondream': 'Moondream',
            'nemotron': 'NeMoVision',
            'llama90b': 'Llama 90B Vision',
            'llama11b': 'Llama 11B Vision'
        }.get(self.model_type, self.model_type)

        print(f"\n🎥 Real-time analyzer started")
        print(f"📁 Session: {self.session_name}")
        print(f"🤖 Model: {model_display} ({'Multimodal LLM' if self.supports_qa() else 'VLM'})")
        print(f"⏱️  Interval: {self.frame_interval}s")
        print(f"🎞️  Record FPS: {self._record_fps}\n")

        return self.session_name

    def stop(self) -> str:
        self.is_running = False

        # Stop analysis thread
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2)

        # Close ffmpeg pipe – this finalises and flushes the MP4
        with self._frames_written_lock:
            total = self._frames_written
        duration = total / self._record_fps
        print(f"📹 Closing pipe – frames written: {total} | approx duration: {duration:.1f}s")

        self._close_ffmpeg_pipe()

        # Validate output file
        if self._saved_video_path and os.path.exists(self._saved_video_path):
            size = os.path.getsize(self._saved_video_path)
            if size < 4096:
                print(f"⚠️ Output MP4 too small ({size} bytes) – discarding")
                try:
                    os.remove(self._saved_video_path)
                except Exception:
                    pass
                self._saved_video_path = None
            else:
                print(f"✅ Saved: {self._saved_video_path} ({size / 1024 / 1024:.1f} MB)")
        else:
            print("⚠️ No output file found after stop()")
            self._saved_video_path = None

        # Update caption records with final video path
        if self._saved_video_path:
            for cap_data in self.captions_data:
                cap_data['video_path'] = self._saved_video_path
            if self.memory and self.captions_data:
                self._update_memory_video_paths()

        self.cleanup_temp_files()

        if self.api_error:
            print(f"⚠️ Stopped (API error) – {self.processed_count} captions")
        else:
            print(f"✅ Stopped – {self.processed_count} captions")

        return self.session_name

    # ============================================================================
    # FRAME INGESTION
    # ============================================================================

    def add_frame(self, frame: np.ndarray):
        """
        Accept a raw webcam frame.

        1. Resize to exact target dimensions.
        2. Throttle-write raw BGR bytes into ffmpeg stdin.
        3. Push a copy to the analysis queue.
        """
        # Always resize to exact target size
        processed = cv2.resize(frame, self._frame_size,
                               interpolation=cv2.INTER_LINEAR)

        # Ensure correct dtype
        if processed.dtype != np.uint8:
            processed = processed.astype(np.uint8)

        now = time.time()
        if now - self._last_write_time >= self._min_write_interval:
            with self._ffmpeg_lock:
                if self._ffmpeg_proc is not None:
                    try:
                        # Write exactly W*H*3 bytes per frame
                        self._ffmpeg_proc.stdin.write(processed.tobytes())
                        self._last_write_time = now
                        with self._frames_written_lock:
                            self._frames_written += 1
                    except BrokenPipeError:
                        # ffmpeg exited unexpectedly
                        print("⚠️ ffmpeg pipe broken – recording stopped")
                        self._ffmpeg_proc = None
                    except Exception as e:
                        print(f"⚠️ Frame write error: {e}")
                        self._ffmpeg_proc = None

        # Queue management for analysis thread
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(processed.copy())

    def _get_video_timestamp(self) -> float:
        """Seconds elapsed in the recorded video based on frames written."""
        with self._frames_written_lock:
            return self._frames_written / self._record_fps

    # ============================================================================
    # MEMORY / DATABASE
    # ============================================================================

    def _update_memory_video_paths(self):
        if not self.memory or not self._saved_video_path:
            return
        try:
            collection = self.memory.collection
            results = collection.get(
                where={"video_source": self.session_name},
                include=["metadatas"]
            )
            if results and results['ids']:
                for idx, item_id in enumerate(results['ids']):
                    meta = results['metadatas'][idx]
                    meta['video_path'] = self._saved_video_path
                    collection.update(ids=[item_id], metadatas=[meta])
                print(f"✅ Updated {len(results['ids'])} DB entries -> "
                      f"{Path(self._saved_video_path).name}")
        except Exception as e:
            print(f"⚠️ DB path update error: {e}")

    def get_saved_video_path(self) -> Optional[str]:
        """Return path to the saved H.264 MP4 (available after stop())."""
        return self._saved_video_path

    # ============================================================================
    # BACKGROUND ANALYSIS THREAD
    # ============================================================================

    def _process_frames(self):
        while self.is_running:
            try:
                if self.api_error:
                    print("\n⚠️ API error – stopping analysis")
                    self.is_running = False
                    break

                current_time = time.time()

                if (current_time - self.last_analysis_time >= self.frame_interval
                        and not self._frame_buffer.is_empty()):
                    self._analyze_buffer(current_time)
                    self._frame_buffer.clear()
                    self.last_analysis_time = current_time

                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    self._frame_buffer.push(frame)
                except queue.Empty:
                    continue

            except Exception as e:
                print(f"❌ Thread error: {e}")
                time.sleep(0.1)

    def _analyze_buffer(self, current_time: float):
        latest_frame = self._frame_buffer.latest()
        if latest_frame is None:
            return

        try:
            caption = self.generate_caption(latest_frame, current_time)

            if caption:
                video_ts = self._get_video_timestamp()

                caption_data = self.create_caption_data(
                    caption=caption,
                    timestamp=current_time,
                    frame_count=self.processed_count,
                    video_source=self.session_name,
                    video_path='webcam_live'
                )

                caption_data['elapsed_time'] = video_ts
                caption_data['timestamp_display'] = self.format_timestamp(video_ts)
                caption_data['elapsed_seconds'] = video_ts
                caption_data['timestamp_value'] = video_ts

                self.captions_data.append(caption_data)
                self.processed_count += 1

                if self.memory:
                    try:
                        self.memory.store_caption_realtime(
                            caption_data,
                            self.session_name,
                            'webcam_live'
                        )
                    except Exception as e:
                        print(f"⚠️ DB store error: {e}")

                display = f"[{caption_data['timestamp_display']}] {caption}"
                self.current_caption = display
                self.caption_queue.put(display)

                if self.on_new_caption:
                    self.on_new_caption(caption, caption_data['timestamp_display'])

                print(f"📸 video_ts={video_ts:.1f}s "
                      f"[{self.format_timestamp(video_ts)}] | {caption[:60]}...")

            if self.api_error:
                self.is_running = False

        except Exception as e:
            print(f"❌ Buffer analysis error: {e}")
            self.stats['errors'] += 1

    # ============================================================================
    # STATUS AND METADATA ACCESSORS
    # ============================================================================

    def get_caption(self) -> str:
        try:
            return self.caption_queue.get_nowait()
        except queue.Empty:
            return self.current_caption

    def get_captions_data(self) -> List[Dict]:
        return self.captions_data

    def get_session_name(self) -> str:
        return self.session_name

    def get_stats(self) -> Dict:
        base_stats = super().get_stats()
        elapsed = time.time() - self.start_time if self.start_time else 0

        with self._frames_written_lock:
            fw = self._frames_written

        model_display = {
            'gemini': 'Gemini',
            'moondream': 'Moondream',
            'nemotron': 'NeMoVision',
            'llama90b': 'Llama 90B Vision',
            'llama11b': 'Llama 11B Vision'
        }.get(self.model_type, self.model_type)

        return {
            **base_stats,
            'session_name': self.session_name,
            'elapsed_time': elapsed,
            'elapsed_display': self.format_timestamp(elapsed),
            'frames_written': fw,
            'video_duration': fw / self._record_fps,
            'queue_size': self.frame_queue.qsize(),
            'buffer_size': self._frame_buffer.size,
            'buffer_capacity': self._frame_buffer.capacity,
            'model_type': model_display,
            'supports_qa': self.supports_qa(),
            'is_running': self.is_running,
            'api_error': self.api_error
        }

    def update_frame_interval(self, new_interval: int):
        self.frame_interval = new_interval
        print(f"⏱️ Frame interval -> {new_interval}s")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_cameras(max_check: int = 5) -> List[int]:
    """Check for available camera devices - ONLY CAMERA 0."""
    available = []
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        available.append(0)
        cap.release()
    return available


def display_webcam_feed(st, analyzer, placeholder):
    """Display webcam feed with real-time analysis in Streamlit."""
    if not analyzer or not analyzer.is_running:
        return

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open camera 0")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        caption_placeholder = st.empty()

        while st.session_state.get('realtime_active', False) and not analyzer.has_api_error():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to grab frame")
                break

            analyzer.add_frame(frame)
            caption = analyzer.get_caption()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            with caption_placeholder.container():
                stats = analyzer.get_stats()
                if analyzer.has_api_error():
                    st.error("⚠️ API error – recording stopped")
                    st.session_state.realtime_active = False
                    break
                else:
                    qa_badge = ("✅ Q&A Ready" if stats.get('supports_qa', False)
                                else "📝 Caption Only")
                    st.info(
                        f"**🎥 Live – Session: {stats['session_name']} | "
                        f"Model: {stats['model_type']} ({qa_badge}) | "
                        f"Captures: {stats['captions_generated']} | "
                        f"Time: {stats['elapsed_display']}**\n\n"
                        f"**🎯 {caption}**"
                    )

            time.sleep(0.03)

    except Exception as e:
        st.error(f"❌ Webcam error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()