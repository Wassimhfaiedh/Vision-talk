"""
API routes for Vision-Talk.
Handles configuration, statistics, search, Q&A, and API key management.
"""

from flask import Blueprint, request, jsonify, session, current_app
from modules.database import (
    get_current_user, login_required, update_user_username,
    update_user_api_keys
)
from utils import get_memory, validate_api_key, get_current_watcher, set_current_model_state, get_processing_jobs
from modules.watcher import VideoWatcher
from config import Config
from routes.video_routes import resolve_video_path

api_bp = Blueprint('api', __name__)
config_obj = Config()


@api_bp.route('/api/update_username', methods=['POST'])
@login_required
def api_update_username():
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


@api_bp.route('/update_api_key', methods=['POST'])
@login_required
def update_api_key():
    user = get_current_user()
    data = request.json
    model_name = data.get('model_name')
    api_key = data.get('api_key')
    
    if not model_name or not api_key:
        return jsonify({'error': 'Model name and API key are required'}), 400
    
    is_valid, error_msg = validate_api_key(model_name, api_key)
    
    if not is_valid:
        return jsonify({'error': error_msg}), 400
    
    current_keys = user['api_keys']
    current_keys[model_name] = api_key
    
    if update_user_api_keys(user['id'], current_keys):
        return jsonify({'success': True, 'message': f'API key validated and saved for {model_name}'})
    else:
        return jsonify({'error': 'Failed to save API key'}), 500


@api_bp.route('/api/validate_api_key', methods=['POST'])
@login_required
def api_validate_api_key():
    """Validate API key without saving it."""
    data = request.json
    model_name = data.get('model_name')
    api_key = data.get('api_key')
    
    if not model_name or not api_key:
        return jsonify({'valid': False, 'error': 'Model name and API key are required'}), 400
    
    is_valid, error_msg = validate_api_key(model_name, api_key)
    
    return jsonify({'valid': is_valid, 'error': error_msg if not is_valid else None})


@api_bp.route('/api/check_model_key', methods=['POST'])
@login_required
def check_model_key():
    """Check if user has an API key saved for a specific model."""
    user = get_current_user()
    data = request.json
    model_name = data.get('model_name')
    
    has_key = model_name in user['api_keys'] and user['api_keys'][model_name]
    return jsonify({'has_key': has_key})


@api_bp.route('/api/config')
@login_required
def get_config():
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


@api_bp.route('/api/stats')
@login_required
def get_stats():
    memory = get_memory()
    if not memory:
        return jsonify({'error': 'Memory not initialized'}), 503
    return jsonify(memory.get_collection_stats())


@api_bp.route('/api/sources')
@login_required
def get_sources():
    memory = get_memory()
    if not memory:
        return jsonify({'error': 'Memory not initialized'}), 503
    return jsonify({'sources': memory.get_video_sources()})


@api_bp.route('/api/reset', methods=['POST'])
@login_required
def reset_database():
    memory = get_memory()
    if memory:
        memory.reset_database()
    return jsonify({'success': True})


@api_bp.route('/api/search', methods=['POST'])
@login_required
def search():
    memory = get_memory()
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
            session_name = result.get('video_source', '')
            resolved = resolve_video_path(session_name)
            if resolved:
                result['video_path'] = str(resolved)

    return jsonify(results)


@api_bp.route('/api/qa', methods=['POST'])
@login_required
def ask_question():
    memory = get_memory()
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
            'answer': "Moondream does not support Q&A. Please use Gemini Flash 3, NeMoVision, or Llama Vision models.",
            'segments': [], 'segments_analyzed': 0, 'error': True
        })

    current_watcher, current_model_loaded, current_model_type = get_current_watcher()
    need_reload = (current_watcher is None or not current_model_loaded or current_model_type != requested_model_type)

    if need_reload:
        print(f"🔄 Loading {model_name} for Q&A...")
        new_watcher = VideoWatcher(config_obj)
        model_loaded = False

        if requested_model_type == 'gemini':
            if api_key:
                model_loaded = new_watcher.load_gemini_model(api_key)
            else:
                return jsonify({'answer': "Gemini API key is required.", 'segments': [], 'segments_analyzed': 0, 'error': True})
        elif requested_model_type == 'nemotron':
            if nvidia_api_key:
                model_loaded = new_watcher.load_nvidia_model('nemotron', nvidia_api_key)
            else:
                return jsonify({'answer': "NVIDIA API key is required for NeMoVision.", 'segments': [], 'segments_analyzed': 0, 'error': True})
        elif requested_model_type == 'llama90b':
            if nvidia_api_key:
                model_loaded = new_watcher.load_nvidia_model('llama90b', nvidia_api_key)
            else:
                return jsonify({'answer': "NVIDIA API key is required for Llama 90B Vision.", 'segments': [], 'segments_analyzed': 0, 'error': True})
        elif requested_model_type == 'llama11b':
            if nvidia_api_key:
                model_loaded = new_watcher.load_nvidia_model('llama11b', nvidia_api_key)
            else:
                return jsonify({'answer': "NVIDIA API key is required for Llama 11B Vision.", 'segments': [], 'segments_analyzed': 0, 'error': True})

        if model_loaded and new_watcher.supports_qa():
            set_current_model_state(new_watcher, requested_model_type, True)
            print(f"✅ Successfully loaded {model_name}")
        else:
            set_current_model_state(None, None, False)
            return jsonify({
                'answer': f"Failed to load {model_name}. Please check your API key.",
                'segments': [], 'segments_analyzed': 0, 'error': True
            })
    
    current_watcher, current_model_loaded, current_model_type = get_current_watcher()
    context_data = memory.prepare_for_qa(question, n_results=n_segments)
    results = context_data.get('results', [])

    if not results:
        return jsonify({
            'answer': 'No relevant video segments found.',
            'segments': [], 'segments_analyzed': 0, 'error': True
        })

    for result in results:
        if result.get('video_path') == 'webcam_live':
            session_name = result.get('video_source', '')
            resolved = resolve_video_path(session_name)
            if resolved:
                result['video_path'] = str(resolved)

    segments_info = []
    for i, result in enumerate(results, 1):
        timestamp = result.get('timestamp_display', '00:00:00')
        video_source = result.get('video_source', 'Unknown')
        score = result.get('score', 0)
        score_color = '#10b981' if score >= 0.7 else '#f59e0b' if score >= 0.4 else '#ef4444'
        segments_info.append({
            'id': i, 'timestamp': timestamp, 'video_source': video_source,
            'score': score, 'score_color': score_color,
            'caption': result.get('caption', ''), 'video_path': result.get('video_path', '')
        })

    try:
        if current_watcher and current_model_loaded:
            print(f"🤖 Using {current_model_type} to answer question...")
            llm_answer = current_watcher.answer_question(question, context_data)

            if llm_answer and not llm_answer.get('error'):
                return jsonify({
                    'answer': llm_answer.get('answer', 'No answer generated'),
                    'segments': segments_info, 'segments_analyzed': len(results),
                    'model_used': llm_answer.get('model', model_name), 'error': False
                })
            else:
                error_msg = llm_answer.get('answer', 'Unknown error') if llm_answer else 'Failed to generate answer'
                return jsonify({
                    'answer': f"Error: {error_msg}", 'segments': segments_info,
                    'segments_analyzed': len(results), 'error': True
                })
        else:
            return jsonify({
                'answer': f"No model loaded. Please select {model_name} and add API key.",
                'segments': segments_info, 'segments_analyzed': len(results), 'error': True
            })
    except Exception as e:
        print(f"Q&A exception: {e}")
        return jsonify({
            'answer': f"Error generating answer: {str(e)}",
            'segments': segments_info, 'segments_analyzed': len(results), 'error': True
        })