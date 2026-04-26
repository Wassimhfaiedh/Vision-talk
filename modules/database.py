"""
Database module for Vision-Talk authentication.
SQLite database with a single 'users' table for single-user mode.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session
from functools import wraps
from flask import redirect, url_for, flash
import secrets
import string

DB_PATH = Path(__file__).parent / "vision_talk.db"

def get_db():
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with users table."""
    conn = get_db()
    cursor = conn.cursor()
    
    # Users table with all required fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            reset_code TEXT NOT NULL,
            api_keys TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if email column exists (for migration)
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'email' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
    if 'reset_code' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN reset_code TEXT")
    
    # Password reset codes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_reset_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            code TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def get_user_count() -> int:
    """Get total number of registered users."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM users")
    count = cursor.fetchone()['count']
    conn.close()
    return count

def is_registration_allowed() -> bool:
    """Check if new registration is allowed (database empty)."""
    return get_user_count() == 0

def generate_reset_code() -> str:
    """Generate a 6-digit reset code."""
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def create_user(username: str, password: str, email: str, initial_model: str = None, initial_api_key: str = None) -> dict:
    """Create a new user."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        password_hash = generate_password_hash(password)
        reset_code = generate_reset_code()
        api_keys = {}
        
        if initial_model and initial_api_key:
            api_keys[initial_model] = initial_api_key
        
        api_keys_json = json.dumps(api_keys)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, email, reset_code, api_keys) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, email, reset_code, api_keys_json)
        )
        
        conn.commit()
        user_id = cursor.lastrowid
        
        # Get the created user
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row['id'],
                'username': row['username'],
                'email': row['email'],
                'password_hash': row['password_hash'],
                'reset_code': row['reset_code'],
                'api_keys': json.loads(row['api_keys']),
                'created_at': row['created_at']
            }
        return None
        
    except sqlite3.IntegrityError:
        conn.close()
        return None

def get_user_by_username(username: str) -> dict:
    """Get user by username."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'password_hash': row['password_hash'],
            'reset_code': row['reset_code'],
            'api_keys': json.loads(row['api_keys']),
            'created_at': row['created_at']
        }
    return None

def get_user_by_email(email: str) -> dict:
    """Get user by email."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'password_hash': row['password_hash'],
            'reset_code': row['reset_code'],
            'api_keys': json.loads(row['api_keys']),
            'created_at': row['created_at']
        }
    return None

def get_user_by_id(user_id: int) -> dict:
    """Get user by ID."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'password_hash': row['password_hash'],
            'reset_code': row['reset_code'],
            'api_keys': json.loads(row['api_keys']),
            'created_at': row['created_at']
        }
    return None

def verify_user(username: str, password: str) -> dict:
    """Verify user credentials."""
    user = get_user_by_username(username)
    if user and check_password_hash(user['password_hash'], password):
        return user
    return None

def verify_user_by_code(username: str, code: str) -> dict:
    """Verify user using permanent recovery code."""
    user = get_user_by_username(username)
    if user and user['reset_code'] == code:
        return user
    return None

def update_user_password(user_id: int, new_password: str) -> bool:
    """Update user's password and generate new reset code."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        password_hash = generate_password_hash(new_password)
        new_reset_code = generate_reset_code()
        cursor.execute(
            "UPDATE users SET password_hash = ?, reset_code = ? WHERE id = ?",
            (password_hash, new_reset_code, user_id)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception:
        conn.close()
        return False

def update_user_username(user_id: int, new_username: str) -> bool:
    """Update user's username."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE users SET username = ? WHERE id = ?",
            (new_username, user_id)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except sqlite3.IntegrityError:
        conn.close()
        return False

def update_user_email(user_id: int, new_email: str) -> bool:
    """Update user's email."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE users SET email = ? WHERE id = ?",
            (new_email, user_id)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except sqlite3.IntegrityError:
        conn.close()
        return False

def update_user_api_keys(user_id: int, api_keys: dict) -> bool:
    """Update user's API keys."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        api_keys_json = json.dumps(api_keys)
        cursor.execute(
            "UPDATE users SET api_keys = ? WHERE id = ?",
            (api_keys_json, user_id)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception:
        conn.close()
        return False

def get_api_key_for_model(user_id: int, model_name: str) -> str:
    """Get API key for a specific model from user's stored keys."""
    user = get_user_by_id(user_id)
    if user and model_name in user['api_keys']:
        return user['api_keys'][model_name]
    return None

def save_reset_code(user_id: int, code: str, expires_minutes: int = 10) -> bool:
    """Save a temporary password reset code."""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        expires_at = datetime.now() + timedelta(minutes=expires_minutes)
        cursor.execute(
            "INSERT INTO password_reset_codes (user_id, code, expires_at) VALUES (?, ?, ?)",
            (user_id, code, expires_at)
        )
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception:
        conn.close()
        return False

def verify_reset_code(user_id: int, code: str) -> bool:
    """Verify a temporary password reset code."""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM password_reset_codes WHERE user_id = ? AND code = ? AND expires_at > ?",
        (user_id, code, datetime.now())
    )
    row = cursor.fetchone()
    conn.close()
    
    return row is not None

def login_user(user: dict):
    """Store user in session."""
    session['user_id'] = user['id']
    session['username'] = user['username']

def logout_user():
    """Clear user from session."""
    session.pop('user_id', None)
    session.pop('username', None)

def get_current_user() -> dict:
    """Get current logged-in user from session."""
    if 'user_id' not in session:
        return None
    return get_user_by_id(session['user_id'])

def login_required(f):
    """Decorator to protect routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function