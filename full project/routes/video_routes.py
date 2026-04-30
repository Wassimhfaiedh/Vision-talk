"""
Video processing routes for Vision-Talk.
Handles video upload, processing, frame extraction, and serving.
"""

import os
import cv2
import base64
import threading
import uuid
from pathlib import Path
from flask import Blueprint, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from modules.database import login_required, get_current_user
from modules.watcher import VideoWatcher
from utils import get_memory, get_processing_jobs, add_processing_job, update_processing_job, get_current_watcher, set_current_model_state
from config import Config

video_bp = Blueprint('video', __name__)
config_obj = Config()


def video_search_folders():
    """Get list of folders to search for videos."""
    folders = [config_obj.UPLOAD_FOLDER]
    if hasattr(config_obj, 'RECORDINGS_FOLDER'):
        folders.append(config_obj.RECORDINGS_FOLDER)
    return folders


def resolve_video_path(video_path: str):
    """Resolve video file path from various sources."""
    if not video_path or video_path == 'webcam_live':
        return None

    if os.path.isabs(video_path):
        p = Path(video_path)
        if p.exists():
            return p

    for folder in video_search_folders():
        p = folder / Path(video_path).name
        if p.exists():
            return p

    stem = Path(video_path).stem
    for folder in video_search_folders():
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            p = folder / (stem + ext)
            if p.exists():
                return p

    return None


@video_bp.route('/')
@login_required
def index():
    return render_template('index.html')


@video_bp.route('/api/upload', methods=['POST'])
@login_required
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = config_obj.UPLOAD_FOLDER / filename
    file.save(str(filepath))

    return jsonify({'success': True, 'filename': filename, 'path': str(filepath)})


@video_bp.route('/api/video/<path:filename>')
@login_required
def serve_video(filename):
    if os.path.isabs(filename):
        filepath = Path(filename)
    else:
        filepath = config_obj.UPLOAD_FOLDER / filename

    if filepath.exists():
        return send_file(filepath, mimetype='video/mp4')

    if hasattr(config_obj, 'RECORDINGS_FOLDER'):
        alt = config_obj.RECORDINGS_FOLDER / filename
        if alt.exists():
            return send_file(alt, mimetype='video/mp4')

    stem = Path(filename).stem
    for folder in video_search_folders():
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            candidate = folder / (stem + ext)
            if candidate.exists():
                return send_file(candidate, mimetype='video/mp4')

    return jsonify({'error': 'Video not found'}), 404


@video_bp.route('/api/frame')
@login_required
def get_frame():
    video_path = request.args.get('video_path')
    timestamp = float(request.args.get('timestamp', 0))

    if not video_path:
        return jsonify({'error': 'No video path'}), 400

    if video_path == 'webcam_live':
        return jsonify({'error': 'Live webcam - no file available'}), 404

    filepath = resolve_video_path(video_path)
    if filepath is None:
        return jsonify({'error': 'Video not found'}), 404

    cap = None
    try:
        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video'}), 404

        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()

        if not ret:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_idx = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

        if not ret:
            return jsonify({'error': 'Could not extract frame'}), 404

        height, width = frame.shape[:2]
        if width > 400:
            new_width = 400
            new_height = int(height * (400 / width))
            frame = cv2.resize(frame, (new_width, new_height))

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(buffer).decode()

        return jsonify({'image': frame_b64, 'timestamp': timestamp})
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if cap:
            cap.release()


@video_bp.route('/api/process/start', methods=['POST'])
@login_required
def start_processing():
    user = get_current_user()
    data = request.json
    job_id = str(uuid.uuid4())

    raw_path = data.get('video_path')
    if not raw_path:
        return jsonify({'error': 'No video path provided'}), 400

    resolved = resolve_video_path(raw_path)
    if resolved is None:
        return jsonify({'error': f'Video file not found: {raw_path}'}), 400
    video_path = str(resolved)

    model_name = data.get('model_name', '')
    api_key = user['api_keys'].get(model_name, '')
    
    if not api_key:
        return jsonify({'error': f'No API key found for {model_name}. Please add it in your profile.'}), 400
    
    nvidia_api_key = api_key if model_name in ['NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision'] else ''
    
    model_type_map = {
        'Gemini Flash 3': 'gemini',
        'NeMoVision': 'nemotron',
        'Llama 90B Vision': 'llama90b',
        'Llama 11B Vision': 'llama11b',
        'Moondream': 'moondream'
    }
    current_model_type = model_type_map.get(model_name, 'gemini')
    set_current_model_state(None, current_model_type, False)

    add_processing_job(job_id, {
        'job_id': job_id, 'running': True, 'complete': False,
        'progress': 0, 'captions_saved': 0,
        'status_text': 'Starting processing...', 'error': None,
        'video_path': video_path, 'model_name': model_name,
        'frame_interval': data.get('frame_interval', 5)
    })

    thread = threading.Thread(
        target=_process_video_background,
        args=(job_id, video_path, model_name, data.get('frame_interval', 5), api_key, nvidia_api_key)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'started'})


def _process_video_background(job_id, video_path, model_name, frame_interval, api_key, nvidia_api_key):
    memory = get_memory()
    
    try:
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")

        watcher = VideoWatcher(config_obj)
        model_loaded = False

        if model_name == 'Gemini Flash 3':
            if not api_key:
                raise Exception("Gemini API key is required")
            model_loaded = watcher.load_gemini_model(api_key)
        elif model_name == 'Moondream':
            if not api_key:
                raise Exception("Moondream API key is required")
            model_loaded = watcher.load_moondream_model(api_key)
        elif model_name == 'NeMoVision':
            if not nvidia_api_key:
                raise Exception("NVIDIA API key is required")
            model_loaded = watcher.load_nvidia_model('nemotron', nvidia_api_key)
        elif model_name == 'Llama 90B Vision':
            if not nvidia_api_key:
                raise Exception("NVIDIA API key is required")
            model_loaded = watcher.load_nvidia_model('llama90b', nvidia_api_key)
        elif model_name == 'Llama 11B Vision':
            if not nvidia_api_key:
                raise Exception("NVIDIA API key is required")
            model_loaded = watcher.load_nvidia_model('llama11b', nvidia_api_key)
        else:
            raise Exception(f"Unknown model: {model_name}")

        if not model_loaded:
            raise Exception(f"Failed to load {model_name} model")

        set_current_model_state(watcher, None, True)

        video_name = Path(video_path).name
        video_info = watcher.get_video_info(video_path)
        estimated_frames = max(1, int(video_info.get('duration', 0) / frame_interval)) if video_info else 100

        def on_caption_generated(result):
            if memory and memory.store_caption_realtime(result, video_name, video_path):
                update_processing_job(job_id, 'captions_saved', 1, add=True)
                current = get_processing_jobs().get(job_id, {})
                if current:
                    progress = int((current.get('captions_saved', 0) / estimated_frames) * 100)
                    update_processing_job(job_id, 'progress', min(99, progress))

        watcher.process_video(video_path, frame_interval=frame_interval, on_caption_generated=on_caption_generated)

        if watcher.has_api_error():
            update_processing_job(job_id, 'status_text', "⚠️ API error occurred")
            update_processing_job(job_id, 'error', "API error")
        else:
            update_processing_job(job_id, 'progress', 100)
            update_processing_job(job_id, 'complete', True)
            update_processing_job(job_id, 'status_text', "✅ Processing complete!")
        update_processing_job(job_id, 'running', False)

    except Exception as e:
        error_msg = str(e)
        print(f"Processing error: {error_msg}")
        update_processing_job(job_id, 'running', False)
        update_processing_job(job_id, 'error', error_msg)
        update_processing_job(job_id, 'status_text', f"❌ Error: {error_msg}")


@video_bp.route('/api/process/status/<job_id>')
@login_required
def get_process_status(job_id):
    jobs = get_processing_jobs()
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(jobs[job_id])