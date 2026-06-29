"""
WebSocket routes for Vision-Talk.
Handles real-time webcam streaming and frame capture.
"""

import cv2
import base64
import threading
import time
from flask import Blueprint, session
from flask_socketio import emit
from modules.database import get_user_by_id
from modules.realtime import RealTimeAnalyzer, get_available_cameras
from utils import (
    get_memory, get_webcam_state, set_webcam_state,
    get_current_watcher, set_current_model_state
)
from config import Config

websocket_bp = Blueprint('websocket', __name__)
config_obj = Config()

# Global socketio instance
socketio = None


def init_socketio(socketio_instance):
    """Initialize SocketIO with event handlers."""
    global socketio
    socketio = socketio_instance
    
    @socketio.on('webcam_start')
    def handle_webcam_start(data):
        _handle_webcam_start(data)
    
    @socketio.on('webcam_stop')
    def handle_webcam_stop():
        _handle_webcam_stop()
    
    @socketio.on('webcam_update_interval')
    def handle_webcam_update_interval(data):
        _handle_webcam_update_interval(data)
    
    @socketio.on('get_cameras')
    def handle_get_cameras():
        _handle_get_cameras()


def _handle_webcam_start(data):
    active_webcam, webcam_analyzer, webcam_thread, webcam_cap = get_webcam_state()
    
    if 'user_id' not in session:
        emit('webcam_error', {'error': 'Please login first'})
        return
    
    user = get_user_by_id(session['user_id'])
    
    if active_webcam:
        emit('webcam_error', {'error': 'Webcam already active'})
        return

    model_type = data.get('model', 'Gemini Flash 3')
    interval = data.get('interval', 5)
    camera_id = data.get('camera', 0)

    model_map = {
        'Gemini Flash 3': 'gemini', 'Moondream': 'moondream',
        'NeMoVision': 'nemotron', 'Llama 90B Vision': 'llama90b',
        'Llama 11B Vision': 'llama11b'
    }

    model_key = model_map.get(model_type, 'gemini')
    api_key = user['api_keys'].get(model_type, '')
    
    if not api_key:
        emit('webcam_error', {'error': f'No API key found for {model_type}. Please add it in your profile.'})
        return

    nvidia_api_key = api_key if model_key in ['nemotron', 'llama90b', 'llama11b'] else ''

    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            emit('webcam_error', {'error': f'Cannot open camera {camera_id}'})
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)

        memory = get_memory()
        analyzer = RealTimeAnalyzer(
            config=config_obj, model_type=model_key,
            api_key=api_key if model_key in ['gemini', 'moondream'] else None,
            nvidia_api_key=nvidia_api_key if model_key in ['nemotron', 'llama90b', 'llama11b'] else None,
            frame_interval=interval, memory=memory
        )

        set_current_model_state(analyzer, model_key, True)

        session_name = analyzer.start()
        set_webcam_state(active=True, analyzer=analyzer, cap=cap)

        emit('webcam_started', {
            'session_name': session_name, 'model': model_type,
            'supports_qa': analyzer.supports_qa()
        })

        thread = threading.Thread(target=_capture_frames, args=(analyzer, cap))
        thread.daemon = True
        thread.start()
        set_webcam_state(thread=thread)

    except Exception as e:
        emit('webcam_error', {'error': str(e)})


def _capture_frames(analyzer, cap):
    frame_skip = 0
    
    while True:
        active, _, _, _ = get_webcam_state()
        if not active:
            break
        
        try:
            ret, frame = cap.read()
            if not ret:
                break

            if analyzer:
                analyzer.add_frame(frame)

                if frame_skip % 2 == 0:
                    caption = analyzer.get_caption()
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_b64 = base64.b64encode(buffer).decode()

                    socketio.emit('webcam_frame', {
                        'image': frame_b64, 'caption': caption, 'timestamp': time.time()
                    })

            frame_skip += 1
            time.sleep(0.033)

        except Exception as e:
            print(f"Frame capture error: {e}")
            break

    saved_path = None
    if analyzer:
        saved_path = analyzer.get_saved_video_path()
        analyzer.stop()

        if saved_path:
            memory = get_memory()
            if memory:
                session_name = analyzer.session_name
                try:
                    collection = memory.collection
                    db_results = collection.get(where={"video_source": session_name}, include=["metadatas"])
                    if db_results and db_results['ids']:
                        for idx, item_id in enumerate(db_results['ids']):
                            meta = db_results['metadatas'][idx]
                            if meta.get('video_path') == 'webcam_live':
                                meta['video_path'] = saved_path
                                collection.update(ids=[item_id], metadatas=[meta])
                        print(f"✅ Patched {len(db_results['ids'])} webcam DB entries")
                except Exception as e:
                    print(f"⚠️ Could not patch webcam DB entries: {e}")

    if cap:
        cap.release()

    set_webcam_state(active=False, analyzer=None, cap=None, thread=None)
    socketio.emit('webcam_stopped', {'saved_video': saved_path})


def _handle_webcam_stop():
    active, analyzer, cap, thread = get_webcam_state()
    
    set_webcam_state(active=False)
    
    saved_path = None
    if analyzer:
        saved_path = analyzer.get_saved_video_path()
        analyzer.stop()
        
        if saved_path:
            emit('webcam_stopped', {'saved_video': saved_path})
            set_webcam_state(analyzer=None)
            return

    if cap:
        cap.release()

    set_webcam_state(analyzer=None, cap=None)
    emit('webcam_stopped', {'saved_video': None})


def _handle_webcam_update_interval(data):
    _, analyzer, _, _ = get_webcam_state()
    if analyzer:
        analyzer.update_frame_interval(data.get('interval', 5))


def _handle_get_cameras():
    cameras = get_available_cameras(5)
    emit('cameras_list', {'cameras': cameras})