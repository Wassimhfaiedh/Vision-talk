"""
Flask + SocketIO main application for Vision-Talk.
Serves the web interface and handles real-time communication.
"""

import os
import sys
import json
import cv2
import base64
import threading
import time
import uuid
import secrets
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, send_file, 
    flash, redirect, url_for, session
)
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

sys.path.append(str(Path(__file__).parent))

from config import Config
from modules.memory import VectorMemory
from modules.realtime import RealTimeAnalyzer, get_available_cameras
from modules.watcher import VideoWatcher
from modules.database import (
    init_db, get_current_user, login_required, login_user, logout_user,
    verify_user, create_user, get_user_by_id, update_user_password,
    update_user_api_keys, get_api_key_for_model, get_user_by_email,
    verify_user_by_code, save_reset_code, verify_reset_code
)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vision-talk-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize database
init_db()

# Configuration
config = Config()
memory = None

# Global state for processing jobs
processing_jobs = {}
job_lock = threading.Lock()

# Webcam state
active_webcam = False
webcam_analyzer = None
webcam_thread = None
webcam_cap = None

# Store current watcher for Q&A
current_watcher = None
watcher_lock = threading.Lock()
current_model_type = None
current_model_loaded = False

# ============================================================================
# EMAIL TEMPLATE FUNCTIONS - Integrated directly
# ============================================================================

def get_registration_email_html(username: str, code: str, email: str) -> str:
    """HTML template for registration email with permanent recovery code."""
    return render_template('email_templates.html', 
                          template_name='registration',
                          username=username, 
                          code=code, 
                          email=email)

def get_reset_code_email_html(username: str, code: str, email: str) -> str:
    """HTML template for password reset email with temporary code."""
    return render_template('email_templates.html',
                          template_name='reset_code',
                          username=username,
                          code=code,
                          email=email)

def get_new_code_email_html(username: str, code: str, email: str) -> str:
    """HTML template for email with new permanent code after password change."""
    return render_template('email_templates.html',
                          template_name='new_code',
                          username=username,
                          code=code,
                          email=email)

# ============================================================================
# EMAIL SENDING FUNCTIONS
# ============================================================================

# EMAIL CONFIGURATION
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "supportvisiontalk@gmail.com"  
SMTP_PASSWORD = "oklt eduk lgqb khce"  

def send_html_email(to_email: str, subject: str, html_content: str, text_content: str = None):
    """Send HTML email using SMTP."""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        if text_content:
            part_text = MIMEText(text_content, 'plain')
            msg.attach(part_text)
        
        part_html = MIMEText(html_content, 'html')
        msg.attach(part_html)
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_EMAIL, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent to {to_email}")
        return True
        
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

def generate_reset_code() -> str:
    """Generate a 6-digit reset code."""
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def send_registration_email(to_email: str, code: str, username: str):
    """Send registration email."""
    subject = f"🔐 Vision-Talk - Your Permanent Recovery Code"
    html_content = get_registration_email_html(username, code, to_email)
    text_content = f"Hello {username},\n\nYour permanent recovery code is: {code}"
    send_html_email(to_email, subject, html_content, text_content)

def send_reset_code_email(to_email: str, code: str, username: str):
    """Send password reset email."""
    subject = f"🔐 Vision-Talk - Password Reset Code"
    html_content = get_reset_code_email_html(username, code, to_email)
    text_content = f"Hello {username},\n\nYour temporary reset code is: {code}\nExpires in 10 minutes."
    send_html_email(to_email, subject, html_content, text_content)

def send_new_code_email(to_email: str, code: str, username: str):
    """Send new recovery code email."""
    subject = f"🔐 Vision-Talk - Your New Recovery Code"
    html_content = get_new_code_email_html(username, code, to_email)
    text_content = f"Hello {username},\n\nYour NEW permanent recovery code is: {code}"
    send_html_email(to_email, subject, html_content, text_content)

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    from modules.database import is_registration_allowed
    
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password_or_code = request.form.get('password')
        
        if not username or not password_or_code:
            flash('Username and password/code are required', 'error')
            return render_template('login.html')
        
        user = verify_user(username, password_or_code)
        
        if not user:
            user = verify_user_by_code(username, password_or_code)
        
        if user:
            login_user(user)
            from werkzeug.security import check_password_hash
            if check_password_hash(user['password_hash'], password_or_code):
                flash(f'Welcome back, {username}!', 'success')
            else:
                flash(f'Welcome back, {username}! (login with recovery code)', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password/code', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page - only accessible when database is empty."""
    from modules.database import is_registration_allowed, get_user_count
    
    # If user is already logged in, redirect to index
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    # If database already has a user, redirect to login
    if not is_registration_allowed():
        flash('Registration is closed. Only one account is allowed.', 'warning')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        initial_model = request.form.get('initial_model')
        api_key = request.form.get('api_key')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('register.html')
        
        if not email:
            flash('Email is required for account recovery', 'error')
            return render_template('register.html')
        
        if not initial_model or not api_key:
            flash('Please select a model and enter your API key', 'error')
            return render_template('register.html')
        
        user = create_user(username, password, email, initial_model, api_key)
        
        if user:
            send_registration_email(email, user['reset_code'], username)
            login_user(user)
            flash(f'✅ Account created! A permanent recovery code has been sent to {email}', 'success')
            return redirect(url_for('index'))
        else:
            flash('Username or email already exists', 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page - allows editing username, email, password, API keys."""
    from modules.database import update_user_username, update_user_email, update_user_password, update_user_api_keys
    
    user = get_current_user()
    models = ['Gemini Flash 3', 'Moondream', 'NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision']
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        # Handle username update
        if action == 'update_username':
            new_username = request.form.get('new_username')
            if new_username and new_username != user['username']:
                if update_user_username(user['id'], new_username):
                    session['username'] = new_username
                    flash('Username updated successfully!', 'success')
                else:
                    flash('Username already exists or invalid', 'error')
            return redirect(url_for('profile'))
        
        # Handle email update
        elif action == 'update_email':
            new_email = request.form.get('new_email')
            if new_email and new_email != user['email']:
                if update_user_email(user['id'], new_email):
                    flash('Email updated successfully!', 'success')
                else:
                    flash('Email already exists or invalid', 'error')
            return redirect(url_for('profile'))
        
        # Handle password update
        elif action == 'update_password':
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            
            if not new_password or new_password != confirm_password:
                flash('Passwords do not match', 'error')
            elif len(new_password) < 4:
                flash('Password must be at least 4 characters', 'error')
            else:
                if update_user_password(user['id'], new_password):
                    updated_user = get_user_by_id(user['id'])
                    if updated_user and updated_user.get('email'):
                        send_new_code_email(updated_user['email'], updated_user['reset_code'], updated_user['username'])
                    flash('Password updated successfully! A new recovery code has been sent to your email.', 'success')
                else:
                    flash('Failed to update password', 'error')
            return redirect(url_for('profile'))
        
        # Handle API key update
        elif action == 'update_api_key':
            model_name = request.form.get('model_name')
            api_key = request.form.get('api_key')
            
            if model_name and api_key:
                current_keys = user['api_keys']
                current_keys[model_name] = api_key
                if update_user_api_keys(user['id'], current_keys):
                    flash(f'API key saved for {model_name}', 'success')
                else:
                    flash('Failed to save API key', 'error')
            return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user, models=models)

@app.route('/api/update_username', methods=['POST'])
@login_required
def api_update_username():
    """API endpoint to update username."""
    from modules.database import update_user_username
    
    user = get_current_user()
    data = request.json
    new_username = data.get('username')
    
    if not new_username:
        return jsonify({'error': 'Username is required'}), 400
    
    if update_user_username(user['id'], new_username):
        session['username'] = new_username
        return jsonify({'success': True, 'username': new_username})
    else:
        return jsonify({'error': 'Username already exists or invalid'}), 400

@app.route('/api/update_email', methods=['POST'])
@login_required
def api_update_email():
    """API endpoint to update email."""
    from modules.database import update_user_email
    
    user = get_current_user()
    data = request.json
    new_email = data.get('email')
    
    if not new_email:
        return jsonify({'error': 'Email is required'}), 400
    
    if update_user_email(user['id'], new_email):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Email already exists or invalid'}), 400

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    """Change user password (legacy endpoint)."""
    user = get_current_user()
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not new_password or new_password != confirm_password:
        flash('Passwords do not match', 'error')
        return redirect(url_for('profile'))
    
    if update_user_password(user['id'], new_password):
        updated_user = get_user_by_id(user['id'])
        if updated_user and updated_user.get('email'):
            send_new_code_email(updated_user['email'], updated_user['reset_code'], updated_user['username'])
        flash('Password updated successfully! A new recovery code has been sent to your email.', 'success')
    else:
        flash('Failed to update password', 'error')
    
    return redirect(url_for('profile'))

@app.route('/update_api_key', methods=['POST'])
@login_required
def update_api_key():
    """Update API key for a specific model."""
    user = get_current_user()
    data = request.json
    model_name = data.get('model_name')
    api_key = data.get('api_key')
    
    if not model_name or not api_key:
        return jsonify({'error': 'Model name and API key are required'}), 400
    
    current_keys = user['api_keys']
    current_keys[model_name] = api_key
    
    if update_user_api_keys(user['id'], current_keys):
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Failed to save API key'}), 500

@app.route('/api/check_model_key', methods=['POST'])
@login_required
def check_model_key():
    """Check if user has API key for a specific model."""
    user = get_current_user()
    data = request.json
    model_name = data.get('model_name')
    
    has_key = model_name in user['api_keys'] and user['api_keys'][model_name]
    return jsonify({'has_key': has_key})

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    """Page to request password reset code."""
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        
        if email:
            user = get_user_by_email(email)
            
            if user:
                temp_code = generate_reset_code()
                save_reset_code(user['id'], temp_code, expires_minutes=10)
                send_reset_code_email(user['email'], temp_code, user['username'])
                session['reset_user_id'] = user['id']
                session.pop('code_verified', None)
                flash('A temporary code has been sent to your email.', 'success')
                return redirect(url_for('reset_with_code'))
            else:
                flash('If this email exists, you will receive a code.', 'info')
                return redirect(url_for('login'))
    
    return render_template('forgot_password.html')

@app.route('/reset-with-code', methods=['GET', 'POST'])
def reset_with_code():
    """Page to enter reset code and new password."""
    if 'reset_user_id' not in session:
        flash('Session expired. Please try again.', 'error')
        return redirect(url_for('forgot_password'))
    
    user_id = session['reset_user_id']
    user = get_user_by_id(user_id)
    
    if not user:
        flash('User not found. Please try again.', 'error')
        session.pop('reset_user_id', None)
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if code and not new_password:
            if verify_reset_code(user_id, code):
                session['code_verified'] = True
                flash('Code verified! Enter your new password.', 'success')
                return render_template('reset_with_code.html', step='password', email=user['email'])
            else:
                flash('Invalid or expired code', 'error')
                return render_template('reset_with_code.html', step='code', email=user['email'])
        
        elif new_password and confirm_password:
            if not session.get('code_verified'):
                flash('Please verify your code first', 'error')
                return redirect(url_for('forgot_password'))
            
            if new_password != confirm_password:
                flash('Passwords do not match', 'error')
                return render_template('reset_with_code.html', step='password', email=user['email'])
            
            if len(new_password) < 4:
                flash('Password must be at least 4 characters', 'error')
                return render_template('reset_with_code.html', step='password', email=user['email'])
            
            if update_user_password(user_id, new_password):
                updated_user = get_user_by_id(user_id)
                new_permanent_code = updated_user['reset_code']
                send_new_code_email(user['email'], new_permanent_code, user['username'])
                
                session.pop('reset_user_id', None)
                session.pop('code_verified', None)
                
                flash('Password reset! A new permanent code has been sent to your email.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Error during password reset', 'error')
                return render_template('reset_with_code.html', step='password', email=user['email'])
        
        else:
            flash('Please enter the verification code', 'error')
            return render_template('reset_with_code.html', step='code', email=user['email'])
    
    if session.get('code_verified'):
        return render_template('reset_with_code.html', step='password', email=user['email'])
    else:
        return render_template('reset_with_code.html', step='code', email=user['email'])
# ============================================================================
# INITIALIZATION
# ============================================================================

def init_memory():
    """Initialize vector memory."""
    global memory
    memory = VectorMemory(config)
    memory.initialize()
    return memory

# ============================================================================
# PROTECTED ROUTES
# ============================================================================

@app.route('/')
@login_required
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/config')
@login_required
def get_config():
    """Get application configuration."""
    return jsonify({
        'models': ['Gemini Flash 3', 'Moondream', 'NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision'],
        'frame_interval': 5,
        'supports_qa': {
            'Gemini Flash 3': True,
            'Moondream': False,
            'NeMoVision': True,
            'Llama 90B Vision': True,
            'Llama 11B Vision': True
        }
    })

@app.route('/api/stats')
@login_required
def get_stats():
    """Get database statistics."""
    if not memory:
        return jsonify({'error': 'Memory not initialized'}), 503
    return jsonify(memory.get_collection_stats())

@app.route('/api/sources')
@login_required
def get_sources():
    """Get all video sources."""
    if not memory:
        return jsonify({'error': 'Memory not initialized'}), 503
    return jsonify({'sources': memory.get_video_sources()})

@app.route('/api/reset', methods=['POST'])
@login_required
def reset_database():
    """Reset the database."""
    global memory
    if memory:
        memory.reset_database()
    return jsonify({'success': True})

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_video():
    """Upload a video file."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = config.UPLOAD_FOLDER / filename
    file.save(str(filepath))

    return jsonify({
        'success': True,
        'filename': filename,
        'path': str(filepath)
    })

@app.route('/api/video/<path:filename>')
@login_required
def serve_video(filename):
    """Serve a video file."""
    if os.path.isabs(filename):
        filepath = Path(filename)
    else:
        filepath = config.UPLOAD_FOLDER / filename

    if filepath.exists():
        return send_file(filepath, mimetype='video/mp4')

    if hasattr(config, 'RECORDINGS_FOLDER'):
        alt = config.RECORDINGS_FOLDER / filename
        if alt.exists():
            return send_file(alt, mimetype='video/mp4')

    stem = Path(filename).stem
    for folder in _video_search_folders():
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            candidate = folder / (stem + ext)
            if candidate.exists():
                return send_file(candidate, mimetype='video/mp4')

    return jsonify({'error': 'Video not found'}), 404

def _video_search_folders():
    """Return all folders that might contain video files."""
    folders = [config.UPLOAD_FOLDER]
    if hasattr(config, 'RECORDINGS_FOLDER'):
        folders.append(config.RECORDINGS_FOLDER)
    return folders

@app.route('/api/frame')
@login_required
def get_frame():
    """Extract a frame from video at specific timestamp."""
    video_path = request.args.get('video_path')
    timestamp = float(request.args.get('timestamp', 0))

    if not video_path:
        return jsonify({'error': 'No video path'}), 400

    if video_path == 'webcam_live':
        return jsonify({'error': 'Live webcam - no file available'}), 404

    filepath = _resolve_video_path(video_path)
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

def _resolve_video_path(video_path: str):
    """Resolve a video path string to an existing Path object."""
    if not video_path or video_path == 'webcam_live':
        return None

    if os.path.isabs(video_path):
        p = Path(video_path)
        if p.exists():
            return p

    for folder in _video_search_folders():
        p = folder / Path(video_path).name
        if p.exists():
            return p

    stem = Path(video_path).stem
    for folder in _video_search_folders():
        for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            p = folder / (stem + ext)
            if p.exists():
                return p

    return None

@app.route('/api/process/start', methods=['POST'])
@login_required
def start_processing():
    """Start background video processing."""
    global current_watcher, current_model_type, current_model_loaded
    
    user = get_current_user()
    data = request.json
    job_id = str(uuid.uuid4())

    raw_path = data.get('video_path')
    if not raw_path:
        return jsonify({'error': 'No video path provided'}), 400

    resolved = _resolve_video_path(raw_path)
    if resolved is None:
        return jsonify({'error': f'Video file not found: {raw_path}'}), 400
    video_path = str(resolved)

    model_name = data.get('model_name', '')
    
    api_key = user['api_keys'].get(model_name, '')
    
    if not api_key:
        return jsonify({'error': f'No API key found for {model_name}. Please add it in your profile.'}), 400
    
    nvidia_api_key = api_key if model_name in ['NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision'] else ''
    
    if model_name == 'Gemini Flash 3':
        current_model_type = 'gemini'
    elif model_name == 'NeMoVision':
        current_model_type = 'nemotron'
    elif model_name == 'Llama 90B Vision':
        current_model_type = 'llama90b'
    elif model_name == 'Llama 11B Vision':
        current_model_type = 'llama11b'
    elif model_name == 'Moondream':
        current_model_type = 'moondream'

    with job_lock:
        processing_jobs[job_id] = {
            'job_id': job_id,
            'running': True,
            'complete': False,
            'progress': 0,
            'captions_saved': 0,
            'status_text': 'Starting processing...',
            'error': None,
            'video_path': video_path,
            'model_name': model_name,
            'frame_interval': data.get('frame_interval', 5)
        }

    thread = threading.Thread(
        target=process_video_background,
        args=(job_id, video_path, model_name, data.get('frame_interval', 5), api_key, nvidia_api_key)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'started'})

def process_video_background(job_id, video_path, model_name, frame_interval, api_key, nvidia_api_key):
    """Background video processing function."""
    global memory, current_watcher, current_model_type, current_model_loaded

    try:
        if not os.path.exists(video_path):
            raise Exception(f"Video file not found: {video_path}")

        watcher = VideoWatcher(config)
        model_loaded = False

        if model_name == 'Gemini Flash 3':
            if not api_key:
                raise Exception("Gemini API key is required")
            model_loaded = watcher.load_gemini_model(api_key)
            current_model_type = 'gemini'
        elif model_name == 'Moondream':
            if not api_key:
                raise Exception("Moondream API key is required")
            model_loaded = watcher.load_moondream_model(api_key)
            current_model_type = 'moondream'
        elif model_name == 'NeMoVision':
            if not nvidia_api_key:
                raise Exception("NVIDIA API key is required")
            model_loaded = watcher.load_nvidia_model('nemotron', nvidia_api_key)
            current_model_type = 'nemotron'
        elif model_name == 'Llama 90B Vision':
            if not nvidia_api_key:
                raise Exception("NVIDIA API key is required")
            model_loaded = watcher.load_nvidia_model('llama90b', nvidia_api_key)
            current_model_type = 'llama90b'
        elif model_name == 'Llama 11B Vision':
            if not nvidia_api_key:
                raise Exception("NVIDIA API key is required")
            model_loaded = watcher.load_nvidia_model('llama11b', nvidia_api_key)
            current_model_type = 'llama11b'
        else:
            raise Exception(f"Unknown model: {model_name}")

        if not model_loaded:
            raise Exception(f"Failed to load {model_name} model")

        current_model_loaded = True
        current_watcher = watcher

        video_name = Path(video_path).name
        video_info = watcher.get_video_info(video_path)
        estimated_frames = max(1, int(video_info.get('duration', 0) / frame_interval)) if video_info else 100

        def on_caption_generated(result):
            if memory and memory.store_caption_realtime(result, video_name, video_path):
                with job_lock:
                    if job_id in processing_jobs:
                        processing_jobs[job_id]['captions_saved'] += 1
                        progress = int((processing_jobs[job_id]['captions_saved'] / estimated_frames) * 100)
                        processing_jobs[job_id]['progress'] = min(99, progress)

        watcher.process_video(video_path, frame_interval=frame_interval, on_caption_generated=on_caption_generated)

        with job_lock:
            if job_id in processing_jobs:
                if watcher.has_api_error():
                    processing_jobs[job_id]['status_text'] = "⚠️ API error occurred"
                    processing_jobs[job_id]['error'] = "API error"
                else:
                    processing_jobs[job_id]['progress'] = 100
                    processing_jobs[job_id]['complete'] = True
                    processing_jobs[job_id]['status_text'] = "✅ Processing complete!"
                processing_jobs[job_id]['running'] = False

    except Exception as e:
        error_msg = str(e)
        print(f"Processing error: {error_msg}")
        with job_lock:
            if job_id in processing_jobs:
                processing_jobs[job_id]['running'] = False
                processing_jobs[job_id]['error'] = error_msg
                processing_jobs[job_id]['status_text'] = f"❌ Error: {error_msg}"

@app.route('/api/process/status/<job_id>')
@login_required
def get_process_status(job_id):
    """Get processing job status."""
    with job_lock:
        if job_id not in processing_jobs:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(processing_jobs[job_id])

@app.route('/api/search', methods=['POST'])
@login_required
def search():
    """Perform semantic search."""
    if not memory:
        return jsonify({'error': 'Memory not initialized'}), 503

    data = request.json
    query = data.get('query', '')
    n_results = data.get('n_results', 10)
    filter_source = data.get('filter_source')

    if not query:
        return jsonify({'error': 'No search query provided'}), 400

    results = memory.search_memory(query, n_results=n_results, filter_source=filter_source)

    for result in results:
        if result.get('video_path') == 'webcam_live':
            session = result.get('video_source', '')
            resolved = _resolve_video_path(session)
            if resolved:
                result['video_path'] = str(resolved)

    return jsonify(results)

@app.route('/api/qa', methods=['POST'])
@login_required
def ask_question():
    """Ask a question about video content."""
    global current_watcher, current_model_loaded, current_model_type, memory
    
    user = get_current_user()

    if not memory:
        return jsonify({'error': 'Memory not initialized'}), 503

    data = request.json
    question = data.get('question', '')
    n_segments = data.get('n_segments', 10)
    model_name = data.get('model_name', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    api_key = user['api_keys'].get(model_name, '')
    nvidia_api_key = api_key if model_name in ['NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision'] else ''

    requested_model_type = None
    if 'Gemini' in model_name:
        requested_model_type = 'gemini'
    elif 'NeMoVision' in model_name:
        requested_model_type = 'nemotron'
    elif 'Llama 90B' in model_name:
        requested_model_type = 'llama90b'
    elif 'Llama 11B' in model_name:
        requested_model_type = 'llama11b'
    elif 'Moondream' in model_name:
        requested_model_type = 'moondream'

    if requested_model_type == 'moondream':
        return jsonify({
            'answer': "Moondream does not support Q&A. Please use Gemini Flash 3, NeMoVision, or Llama Vision models for Q&A.",
            'segments': [],
            'segments_analyzed': 0,
            'error': True
        })

    need_reload = (
        current_watcher is None or
        not current_model_loaded or
        current_model_type != requested_model_type
    )

    if need_reload:
        print(f"🔄 Loading {model_name} for Q&A...")
        new_watcher = VideoWatcher(config)
        model_loaded = False

        if requested_model_type == 'gemini':
            if api_key:
                model_loaded = new_watcher.load_gemini_model(api_key)
            else:
                return jsonify({'answer': "Gemini API key is required. Please add it in your profile.", 'segments': [], 'segments_analyzed': 0, 'error': True})
        elif requested_model_type == 'nemotron':
            if nvidia_api_key:
                model_loaded = new_watcher.load_nvidia_model('nemotron', nvidia_api_key)
            else:
                return jsonify({'answer': "NVIDIA API key is required for NeMoVision. Please add it in your profile.", 'segments': [], 'segments_analyzed': 0, 'error': True})
        elif requested_model_type == 'llama90b':
            if nvidia_api_key:
                model_loaded = new_watcher.load_nvidia_model('llama90b', nvidia_api_key)
            else:
                return jsonify({'answer': "NVIDIA API key is required for Llama 90B Vision. Please add it in your profile.", 'segments': [], 'segments_analyzed': 0, 'error': True})
        elif requested_model_type == 'llama11b':
            if nvidia_api_key:
                model_loaded = new_watcher.load_nvidia_model('llama11b', nvidia_api_key)
            else:
                return jsonify({'answer': "NVIDIA API key is required for Llama 11B Vision. Please add it in your profile.", 'segments': [], 'segments_analyzed': 0, 'error': True})

        if model_loaded and new_watcher.supports_qa():
            current_watcher = new_watcher
            current_model_type = requested_model_type
            current_model_loaded = True
            print(f"✅ Successfully loaded {model_name}")
        else:
            current_model_loaded = False
            return jsonify({
                'answer': f"Failed to load {model_name}. Please check your API key in profile and try again.",
                'segments': [], 'segments_analyzed': 0, 'error': True
            })

    context_data = memory.prepare_for_qa(question, n_results=n_segments)
    results = context_data.get('results', [])

    if not results:
        return jsonify({
            'answer': 'No relevant video segments found to answer your question.',
            'segments': [], 'segments_analyzed': 0, 'error': True
        })

    for result in results:
        if result.get('video_path') == 'webcam_live':
            session = result.get('video_source', '')
            resolved = _resolve_video_path(session)
            if resolved:
                result['video_path'] = str(resolved)

    segments_info = []
    for i, result in enumerate(results, 1):
        timestamp = result.get('timestamp_display', '00:00:00')
        video_source = result.get('video_source', 'Unknown')
        score = result.get('score', 0)
        score_color = '#10b981' if score >= 0.7 else '#f59e0b' if score >= 0.4 else '#ef4444'
        segments_info.append({
            'id': i,
            'timestamp': timestamp,
            'video_source': video_source,
            'score': score,
            'score_color': score_color,
            'caption': result.get('caption', ''),
            'video_path': result.get('video_path', '')
        })

    try:
        if current_watcher and current_model_loaded:
            print(f"🤖 Using {current_model_type} to answer question...")
            llm_answer = current_watcher.answer_question(question, context_data)

            if llm_answer and not llm_answer.get('error'):
                return jsonify({
                    'answer': llm_answer.get('answer', 'No answer generated'),
                    'segments': segments_info,
                    'segments_analyzed': len(results),
                    'model_used': llm_answer.get('model', model_name),
                    'error': False
                })
            else:
                error_msg = llm_answer.get('answer', 'Unknown error') if llm_answer else 'Failed to generate answer'
                return jsonify({
                    'answer': f"Error: {error_msg}",
                    'segments': segments_info,
                    'segments_analyzed': len(results),
                    'error': True
                })
        else:
            return jsonify({
                'answer': f"No model loaded. Please select {model_name} and ensure you have added the API key in your profile.",
                'segments': segments_info,
                'segments_analyzed': len(results),
                'error': True
            })
    except Exception as e:
        print(f"Q&A exception: {e}")
        return jsonify({
            'answer': f"Error generating answer: {str(e)}",
            'segments': segments_info,
            'segments_analyzed': len(results),
            'error': True
        })

# ============================================================================
# WEBCAM REAL-TIME ANALYSIS (SocketIO)
# ============================================================================

@socketio.on('webcam_start')
def handle_webcam_start(data):
    """Start webcam real-time analysis."""
    global active_webcam, webcam_analyzer, webcam_thread, webcam_cap, current_watcher, current_model_type, current_model_loaded
    
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
        'Gemini Flash 3': 'gemini',
        'Moondream': 'moondream',
        'NeMoVision': 'nemotron',
        'Llama 90B Vision': 'llama90b',
        'Llama 11B Vision': 'llama11b'
    }

    model_key = model_map.get(model_type, 'gemini')
    
    api_key = user['api_keys'].get(model_type, '')
    
    if not api_key:
        emit('webcam_error', {'error': f'No API key found for {model_type}. Please add it in your profile.'})
        return

    nvidia_api_key = api_key if model_key in ['nemotron', 'llama90b', 'llama11b'] else ''

    try:
        webcam_cap = cv2.VideoCapture(camera_id)
        if not webcam_cap.isOpened():
            emit('webcam_error', {'error': f'Cannot open camera {camera_id}'})
            return

        webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        webcam_cap.set(cv2.CAP_PROP_FPS, 30)

        webcam_analyzer = RealTimeAnalyzer(
            config=config,
            model_type=model_key,
            api_key=api_key if model_key in ['gemini', 'moondream'] else None,
            nvidia_api_key=nvidia_api_key if model_key in ['nemotron', 'llama90b', 'llama11b'] else None,
            frame_interval=interval,
            memory=memory
        )

        current_watcher = webcam_analyzer
        current_model_type = model_key
        current_model_loaded = True

        session_name = webcam_analyzer.start()
        active_webcam = True

        emit('webcam_started', {
            'session_name': session_name,
            'model': model_type,
            'supports_qa': webcam_analyzer.supports_qa()
        })

        webcam_thread = threading.Thread(target=capture_frames)
        webcam_thread.daemon = True
        webcam_thread.start()

    except Exception as e:
        emit('webcam_error', {'error': str(e)})

def capture_frames():
    """Capture frames from webcam and send to analyzer."""
    global active_webcam, webcam_cap, webcam_analyzer

    frame_skip = 0

    while active_webcam and webcam_cap and webcam_cap.isOpened():
        try:
            ret, frame = webcam_cap.read()
            if not ret:
                break

            if webcam_analyzer:
                webcam_analyzer.add_frame(frame)

                if frame_skip % 2 == 0:
                    caption = webcam_analyzer.get_caption()
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    frame_b64 = base64.b64encode(buffer).decode()

                    socketio.emit('webcam_frame', {
                        'image': frame_b64,
                        'caption': caption,
                        'timestamp': time.time()
                    })

            frame_skip += 1
            time.sleep(0.033)

        except Exception as e:
            print(f"Frame capture error: {e}")
            break

    saved_path = None
    if webcam_analyzer:
        saved_path = webcam_analyzer.get_saved_video_path()
        webcam_analyzer.stop()

        if saved_path and memory:
            session = webcam_analyzer.session_name
            try:
                collection = memory.collection
                db_results = collection.get(
                    where={"video_source": session},
                    include=["metadatas"]
                )
                if db_results and db_results['ids']:
                    for idx, item_id in enumerate(db_results['ids']):
                        meta = db_results['metadatas'][idx]
                        if meta.get('video_path') == 'webcam_live':
                            meta['video_path'] = saved_path
                            collection.update(ids=[item_id], metadatas=[meta])
                    print(f"✅ Patched {len(db_results['ids'])} webcam DB entries → {Path(saved_path).name}")
            except Exception as e:
                print(f"⚠️ Could not patch webcam DB entries: {e}")

        webcam_analyzer = None

    if webcam_cap:
        webcam_cap.release()
        webcam_cap = None

    active_webcam = False
    socketio.emit('webcam_stopped', {'saved_video': saved_path})

@socketio.on('webcam_stop')
def handle_webcam_stop():
    """Stop webcam real-time analysis."""
    global active_webcam, webcam_analyzer, webcam_cap

    active_webcam = False

    if webcam_analyzer:
        saved_path = webcam_analyzer.get_saved_video_path()
        webcam_analyzer.stop()
        webcam_analyzer = None
        if saved_path:
            emit('webcam_stopped', {'saved_video': saved_path})
            return

    if webcam_cap:
        webcam_cap.release()
        webcam_cap = None

    emit('webcam_stopped', {'saved_video': None})

@socketio.on('webcam_update_interval')
def handle_webcam_update_interval(data):
    """Update frame analysis interval."""
    global webcam_analyzer
    if webcam_analyzer:
        interval = data.get('interval', 5)
        webcam_analyzer.update_frame_interval(interval)

@socketio.on('get_cameras')
def handle_get_cameras():
    """Get available camera devices."""
    cameras = get_available_cameras(5)
    emit('cameras_list', {'cameras': cameras})

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    init_memory()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)