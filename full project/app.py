"""
Main application entry point for Vision-Talk.
Initializes Flask, SocketIO, and registers routes.
"""

import os
import secrets
from pathlib import Path
from flask import Flask
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import Config
from modules.database import init_db
from routes import register_blueprints
from utils import init_memory

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_CONTENT_LENGTH'] = None

socketio = SocketIO(app, cors_allowed_origins=os.environ.get('CORS_ORIGINS', '*'), async_mode='threading')

# Initialize database
init_db()

# Initialize config
config = Config()

# Register all blueprints
register_blueprints(app, socketio)

# Initialize memory
init_memory()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, 
                 debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true', 
                 use_reloader=False)