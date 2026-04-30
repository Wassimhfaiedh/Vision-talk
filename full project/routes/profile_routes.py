"""
Profile management routes for Vision-Talk.
Handles username, password, API key updates, and email changes.
"""

from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from modules.database import (
    get_current_user, login_required, update_user_username,
    update_user_password, update_user_api_keys, get_user_by_email, get_db_connection
)
from utils import validate_password, validate_api_key, generate_verification_token, send_email_change_verification

profile_bp = Blueprint('profile', __name__)


@profile_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_current_user()
    models = ['Gemini Flash 3', 'Moondream', 'NeMoVision', 'Llama 90B Vision', 'Llama 11B Vision']
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update_username':
            new_username = request.form.get('new_username')
            if new_username and new_username != user['username']:
                if update_user_username(user['id'], new_username):
                    session['username'] = new_username
                    flash('Username updated successfully!', 'success')
                else:
                    flash('Username already exists or invalid', 'error')
            return redirect(url_for('profile.profile'))
        
        elif action == 'update_password':
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')
            
            is_valid, error_msg = validate_password(new_password, confirm_password)
            
            if not is_valid:
                flash(error_msg, 'error')
            else:
                if update_user_password(user['id'], new_password):
                    flash('✅ Password updated successfully!', 'success')
                else:
                    flash('Failed to update password', 'error')
            return redirect(url_for('profile.profile'))
        
        elif action == 'update_api_key':
            model_name = request.form.get('model_name')
            api_key = request.form.get('api_key')
            
            if model_name and api_key:
                is_valid, error_msg = validate_api_key(model_name, api_key)
                
                if not is_valid:
                    flash(f'❌ {error_msg}', 'error')
                    return redirect(url_for('profile.profile'))
                
                current_keys = user['api_keys']
                current_keys[model_name] = api_key
                if update_user_api_keys(user['id'], current_keys):
                    flash(f'✅ API key validated and saved for {model_name}', 'success')
                else:
                    flash('Failed to save API key', 'error')
            return redirect(url_for('profile.profile'))
    
    return render_template('profile.html', user=user, models=models)


@profile_bp.route('/profile/change-email', methods=['POST'])
@login_required
def change_email_request():
    user = get_current_user()
    new_email = request.form.get('new_email')
    
    if not new_email:
        return jsonify({'success': False, 'error': 'Email is required'})
    
    if new_email == user['email']:
        return jsonify({'success': False, 'error': 'New email is the same as current email'})
    
    existing = get_user_by_email(new_email)
    if existing:
        return jsonify({'success': False, 'error': 'Email already in use'})
    
    token = generate_verification_token()
    
    conn = get_db_connection()
    conn.execute('UPDATE users SET pending_email = ?, email_verification_token = ?, email_verification_expires = ? WHERE id = ?',
                (new_email, token, (datetime.now() + timedelta(hours=24)).isoformat(), user['id']))
    conn.commit()
    conn.close()
    
    send_email_change_verification(new_email, token, user['username'])
    
    return jsonify({'success': True, 'message': 'Verification link sent to your new email!'})


@profile_bp.route('/verify-new-email/<token>')
def verify_new_email(token):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email_verification_token = ?', (token,)).fetchone()
    
    if not user:
        conn.close()
        flash('Invalid or expired verification link.', 'error')
        return redirect(url_for('profile.profile'))
    
    expires = datetime.fromisoformat(user['email_verification_expires']) if user['email_verification_expires'] else None
    if expires and expires < datetime.now():
        conn.close()
        flash('Verification link has expired.', 'error')
        return redirect(url_for('profile.profile'))
    
    new_email = user['pending_email']
    if not new_email:
        conn.close()
        flash('No pending email change', 'error')
        return redirect(url_for('profile.profile'))
    
    conn.execute('UPDATE users SET email = ?, pending_email = NULL, email_verification_token = NULL, email_verification_expires = NULL WHERE id = ?',
                (new_email, user['id']))
    conn.commit()
    conn.close()
    
    flash('✅ Email updated successfully!', 'success')
    return redirect(url_for('profile.profile'))