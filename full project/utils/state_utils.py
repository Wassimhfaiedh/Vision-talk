"""
Global state management for Vision-Talk.
Handles memory, processing jobs, webcam state, and model state.
"""

import threading
from modules.memory import VectorMemory
from config import Config

# Global state variables
_memory = None
_processing_jobs = {}
_job_lock = threading.Lock()

_active_webcam = False
_webcam_analyzer = None
_webcam_thread = None
_webcam_cap = None

_current_watcher = None
_watcher_lock = threading.Lock()
_current_model_type = None
_current_model_loaded = False

_config = Config()


# ============================================================================
# Memory Management
# ============================================================================

def get_memory():
    """Get the global memory instance."""
    return _memory


def set_memory(memory):
    """Set the global memory instance."""
    global _memory
    _memory = memory


def init_memory():
    """Initialize the vector memory."""
    global _memory
    _memory = VectorMemory(_config)
    _memory.initialize()
    return _memory


# ============================================================================
# Processing Jobs Management
# ============================================================================

def get_processing_jobs():
    """Get the processing jobs dictionary."""
    return _processing_jobs


def add_processing_job(job_id, job_data):
    """Add a new processing job."""
    with _job_lock:
        _processing_jobs[job_id] = job_data


def update_processing_job(job_id, key, value, add=False):
    """Update a processing job field."""
    with _job_lock:
        if job_id in _processing_jobs:
            if add and isinstance(_processing_jobs[job_id].get(key), (int, float)):
                _processing_jobs[job_id][key] += value
            else:
                _processing_jobs[job_id][key] = value


def remove_processing_job(job_id):
    """Remove a processing job."""
    with _job_lock:
        if job_id in _processing_jobs:
            del _processing_jobs[job_id]


# ============================================================================
# Webcam State Management
# ============================================================================

def get_webcam_state():
    """Get current webcam state."""
    return _active_webcam, _webcam_analyzer, _webcam_thread, _webcam_cap


def set_webcam_state(active=None, analyzer=None, thread=None, cap=None):
    """Set webcam state variables."""
    global _active_webcam, _webcam_analyzer, _webcam_thread, _webcam_cap
    if active is not None:
        _active_webcam = active
    if analyzer is not None:
        _webcam_analyzer = analyzer
    if thread is not None:
        _webcam_thread = thread
    if cap is not None:
        _webcam_cap = cap


# ============================================================================
# Current Model State Management
# ============================================================================

def get_current_watcher():
    """Get current watcher and model state."""
    return _current_watcher, _current_model_loaded, _current_model_type


def set_current_model_state(watcher, model_type, model_loaded):
    """Set current model state."""
    global _current_watcher, _current_model_type, _current_model_loaded
    if watcher is not None:
        _current_watcher = watcher
    if model_type is not None:
        _current_model_type = model_type
    if model_loaded is not None:
        _current_model_loaded = model_loaded