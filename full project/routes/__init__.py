"""
Routes package initialization.
Registers all blueprints with the Flask app.
"""

from routes.auth_routes import auth_bp
from routes.profile_routes import profile_bp
from routes.api_routes import api_bp
from routes.video_routes import video_bp
from routes.websocket_routes import websocket_bp, init_socketio


def register_blueprints(app, socketio):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(auth_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(websocket_bp)
    
    # Initialize SocketIO handlers
    init_socketio(socketio)