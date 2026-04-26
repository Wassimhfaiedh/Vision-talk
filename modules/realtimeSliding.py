"""
Real-time video analysis module for webcam.
Inherits from BaseVideoAnalyzer with queue and threading management.
Immediate database storage.
Records webcam video and saves as H.264 mp4 on stop.

KEY FIX: timestamps stored in captions are derived from actual frames
written to the VideoWriter (frames_written / record_fps), NOT wall-clock
elapsed time. This guarantees frame extraction lands on the correct frame.
"""

import cv2
import numpy as np
from PIL import Image
import threading
import queue
import time
from datetime import datetime
from typing import Optional, List, Dict, Callable
import os
import subprocess

from modules.base_analyzer import BaseVideoAnalyzer


class RealTimeAnalyzer(BaseVideoAnalyzer):
    """Analyzer for real-time webcam stream."""

    def __init__(self, config, model_type: str = 'moondream',
                 api_key: str = None, frame_interval: int = 5,
                 memory=None):
        """Initialize real-time analyzer."""
        super().__init__(config, mode='realtime')

        self.model_type = model_type.lower()
        self.frame_interval = frame_interval
        self.memory = memory

        self.frame_queue = queue.Queue(maxsize=config.REALTIME_MAX_QUEUE_SIZE)
        self.caption_queue = queue.Queue()
        self.is_running = False
        self.analysis_thread = None

        self.current_caption = f"Initializing {model_type}..."
        self.last_analysis_time = 0
        self.frame_buffer = []
        self.captions_data = []
        self.processed_count = 0

        self.start_time = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"webcam_{self.session_id}"

        self.on_new_caption: Optional[Callable] = None

        # ── Video recording state ────────────────────────────────────────────
        self._video_writer = None
        self._raw_video_path = None
        self._saved_video_path = None

        # The FPS we tell the VideoWriter. We throttle writes to this rate so
        # the file duration == real elapsed time.
        self._record_fps = 20.0
        self._frame_size = (640, 360)

        # Counts how many frames have actually been written to the file.
        # Used to compute the accurate video-file timestamp for each caption.
        self._frames_written = 0
        self._frames_written_lock = threading.Lock()

        # Throttle: minimum seconds between consecutive writes
        self._min_write_interval = 1.0 / self._record_fps
        self._last_write_time = 0.0

        self._load_model(api_key)

    def _load_model(self, api_key: str = None) -> bool:
        """Load the specified model."""
        if self.model_type == 'gemini':
            return self.load_gemini_model(api_key)
        else:
            return self.load_moondream_model(api_key)

    # ============================================================================
    # THREAD MANAGEMENT
    # ============================================================================

    def start(self) -> str:
        """Start the analysis thread."""
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

        # Open VideoWriter (raw AVI, converted to H.264 on stop)
        try:
            raw_path = str(self.config.UPLOAD_FOLDER / f"{self.session_name}_raw.avi")
            self._raw_video_path = raw_path
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self._video_writer = cv2.VideoWriter(
                raw_path, fourcc, self._record_fps, self._frame_size
            )
            if not self._video_writer.isOpened():
                print("⚠️ Could not open video writer – recording disabled")
                self._video_writer = None
        except Exception as e:
            print(f"⚠️ Video writer init error: {e}")
            self._video_writer = None

        self.analysis_thread = threading.Thread(target=self._process_frames)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

        print(f"\n🎥 Real-time analyzer started")
        print(f"📁 Session: {self.session_name}")
        print(f"🤖 Model: {self.model_type}")
        print(f"⏱️  Interval: {self.frame_interval}s")
        print(f"🎞️  Record FPS: {self._record_fps}\n")

        return self.session_name

    def stop(self) -> str:
        """Stop the analysis thread and save recorded video as H.264 mp4."""
        self.is_running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2)

        # Release VideoWriter – flushes all buffered frames to disk
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            with self._frames_written_lock:
                total = self._frames_written
            duration = total / self._record_fps
            print(f"📹 Raw video: {self._raw_video_path}")
            print(f"   Frames written: {total}  |  Video duration: {duration:.1f}s")

        # Convert to H.264 mp4
        self._saved_video_path = self._convert_to_h264(self._raw_video_path)

        # Update all caption entries + ChromaDB with the real video path
        if self._saved_video_path:
            for cap_data in self.captions_data:
                cap_data['video_path'] = self._saved_video_path
            if self.memory and self.captions_data:
                self._update_memory_video_paths()

        self.cleanup_temp_files()

        if self.api_error:
            print(f"\n⚠️ Stopped (API error) – {self.processed_count} captions")
        else:
            print(f"\n✅ Stopped – {self.processed_count} captions")
        if self._saved_video_path:
            print(f"✅ Saved: {self._saved_video_path}")

        return self.session_name

    # ──────────────────────────────────────────────────────────────────────────
    # VIDEO WRITING – throttled so file FPS matches _record_fps exactly
    # ──────────────────────────────────────────────────────────────────────────

    def add_frame(self, frame: np.ndarray):
        """
        Called by the webcam loop for every captured frame.

        1.  Resize to recording resolution.
        2.  Write to VideoWriter – but only if enough time has elapsed since
            the last write.  This keeps the file FPS stable regardless of how
            fast the webcam loop runs.
        3.  Push to analysis queue (unchanged behaviour).
        """
        processed_frame = cv2.resize(frame, self._frame_size)

        # Throttled write: only write when ≥ 1/record_fps seconds have passed
        now = time.time()
        if (self._video_writer is not None
                and self._video_writer.isOpened()
                and (now - self._last_write_time) >= self._min_write_interval):
            self._video_writer.write(processed_frame)
            self._last_write_time = now
            with self._frames_written_lock:
                self._frames_written += 1

        # Queue management for analysis thread
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(processed_frame.copy())

    def _get_video_timestamp(self) -> float:
        """
        Return the current position in the *saved video file* (seconds).
        = frames_written_so_far / record_fps
        This is the timestamp to use for frame extraction later.
        """
        with self._frames_written_lock:
            return self._frames_written / self._record_fps

    # ──────────────────────────────────────────────────────────────────────────

    def _convert_to_h264(self, raw_path: str) -> Optional[str]:
        """Convert raw AVI to H.264 mp4 with ffmpeg."""
        if not raw_path or not os.path.exists(raw_path):
            return None

        # Verify the raw file has content
        if os.path.getsize(raw_path) < 1024:
            print("⚠️ Raw video file is too small – skipping conversion")
            return None

        out_path = str(self.config.UPLOAD_FOLDER / f"{self.session_name}.mp4")

        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', raw_path,
                '-vcodec', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-an',                    # no audio (webcam usually has none)
                '-movflags', '+faststart',
                out_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ H.264 mp4 saved: {out_path}")
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
                return out_path
            else:
                print(f"⚠️ ffmpeg failed: {result.stderr[-400:]}")
                # Fallback: keep the raw AVI
                fallback = str(self.config.UPLOAD_FOLDER / f"{self.session_name}.avi")
                try:
                    os.rename(raw_path, fallback)
                except Exception:
                    pass
                return fallback if os.path.exists(fallback) else None

        except FileNotFoundError:
            print("⚠️ ffmpeg not found – keeping raw AVI")
            fallback = str(self.config.UPLOAD_FOLDER / f"{self.session_name}.avi")
            try:
                os.rename(raw_path, fallback)
            except Exception:
                pass
            return fallback if os.path.exists(fallback) else None
        except Exception as e:
            print(f"⚠️ ffmpeg error: {e}")
            return None

    def _update_memory_video_paths(self):
        """Patch video_path in every ChromaDB entry for this session."""
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
                    # elapsed_seconds is already stored as timestamp_value
                    collection.update(ids=[item_id], metadatas=[meta])
                print(f"✅ Updated {len(results['ids'])} DB entries → {Path(self._saved_video_path).name}")
        except Exception as e:
            print(f"⚠️ DB path update error: {e}")

    def get_saved_video_path(self) -> Optional[str]:
        """Return path to the saved H.264 video (available after stop())."""
        return self._saved_video_path

    # ============================================================================
    # FRAME PROCESSING THREAD
    # ============================================================================

    def _process_frames(self):
        """Main analysis thread."""
        while self.is_running:
            try:
                if self.api_error:
                    print("\n⚠️ API error – stopping analysis")
                    self.is_running = False
                    break

                current_time = time.time()

                if current_time - self.last_analysis_time >= self.frame_interval:
                    if self.frame_buffer:
                        self._analyze_buffer(current_time)
                        self.frame_buffer.clear()
                        self.last_analysis_time = current_time

                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    self.frame_buffer.append(frame)
                    if len(self.frame_buffer) > self.config.REALTIME_MAX_BUFFER_SIZE:
                        self.frame_buffer.pop(0)
                except queue.Empty:
                    continue

            except Exception as e:
                print(f"❌ Thread error: {e}")
                time.sleep(0.1)

    def _analyze_buffer(self, current_time: float):
        """Analyze the latest frame in the buffer and store caption."""
        if not self.frame_buffer:
            return

        try:
            latest_frame = self.frame_buffer[-1]
            caption = self.generate_caption(latest_frame, current_time)

            if caption:
                # video_ts = exact position in the saved video file (seconds).
                # Derived from frames actually written (not wall-clock), so it
                # is always consistent with what extract_frame_cached seeks to
                # AND with what the video player jump uses.
                video_ts = self._get_video_timestamp()

                caption_data = self.create_caption_data(
                    caption=caption,
                    timestamp=current_time,       # absolute epoch kept for DB
                    frame_count=self.processed_count,
                    video_source=self.session_name,
                    video_path='webcam_live'       # replaced on stop()
                )

                # timestamp_display = video-file position so the label shown
                # in the UI always matches where Jump seeks to.
                caption_data['elapsed_time'] = video_ts
                caption_data['timestamp_display'] = self.format_timestamp(video_ts)

                # elapsed_seconds = seek position for extract_frame_cached
                caption_data['elapsed_seconds'] = video_ts

                # timestamp_value in DB carries the same value so search
                # results can resolve it without extra fields
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

                print(f"📸 video_ts={video_ts:.1f}s [{self.format_timestamp(video_ts)}] | {caption[:60]}...")

            if self.api_error:
                self.is_running = False

        except Exception as e:
            print(f"❌ Buffer analysis error: {e}")
            self.stats['errors'] += 1

    # ============================================================================
    # ACCESSORS
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

        return {
            **base_stats,
            'session_name': self.session_name,
            'elapsed_time': elapsed,
            'elapsed_display': self.format_timestamp(elapsed),
            'frames_written': fw,
            'video_duration': fw / self._record_fps,
            'queue_size': self.frame_queue.qsize(),
            'buffer_size': len(self.frame_buffer),
            'model_type': self.model_type,
            'is_running': self.is_running,
            'api_error': self.api_error
        }

    def update_frame_interval(self, new_interval: int):
        self.frame_interval = new_interval
        print(f"⏱️ Frame interval → {new_interval}s")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_available_cameras(max_check=5) -> List[int]:
    """Check for available camera indices."""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def display_webcam_feed(st, analyzer, placeholder):
    """Display webcam feed with real-time analysis."""
    if not analyzer or not analyzer.is_running:
        return

    try:
        camera_idx = st.session_state.get('realtime_camera', 0)
        cap = cv2.VideoCapture(camera_idx)

        if not cap.isOpened():
            st.error(f"❌ Cannot open camera {camera_idx}")
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
                    st.info(
                        f"**🎥 Live – Session: {stats['session_name']} | "
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


# make Path available for _update_memory_video_paths
from pathlib import Path