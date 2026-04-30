"""
Authentication routes for Vision-Talk.
Handles login, registration, email verification, and password reset.
"""

from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from modules.database import (
    get_current_user, login_required, login_user, logout_user,
    verify_user, create_user, get_user_by_id, update_user_password,
    get_user_by_email, verify_user_by_code, save_reset_code, verify_reset_code,
    get_db_connection, is_registration_allowed
)
from utils import (
    generate_verification_token, generate_reset_code,
    send_registration_email, send_reset_code_email, send_new_code_email,
    validate_password
)

auth_bp = Blueprint('auth', __name__)

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('video.index'))
    
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
            if not user.get('email_verified', False):
                flash('Please verify your email before logging in.', 'warning')
                return redirect(url_for('auth.login'))
            
            login_user(user)
            from werkzeug.security import check_password_hash
            if check_password_hash(user['password_hash'], password_or_code):
                flash(f'Welcome back, {username}!', 'success')
            else:
                flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('video.index'))
        else:
            flash('Invalid username or password/code', 'error')
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('video.index'))
    
    if not is_registration_allowed():
        flash('Registration is closed. Only one account is allowed.', 'warning')
        return redirect(url_for('auth.login'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return render_template('register.html')
        
        if not email:
            flash('Email is required for account recovery', 'error')
            return render_template('register.html')
        
        is_valid, error_msg = validate_password(password)
        if not is_valid:
            flash(error_msg, 'error')
            return render_template('register.html')
        
        user = create_user(username, password, email, None, None)
        
        if user:
            token = generate_verification_token()
            recovery_code = user['reset_code']
            
            conn = get_db_connection()
            conn.execute('UPDATE users SET verification_token = ?, verification_token_expires = ?, email_verified = 0 WHERE id = ?',
                        (token, (datetime.now() + timedelta(hours=24)).isoformat(), user['id']))
            conn.commit()
            conn.close()
            
            send_registration_email(email, recovery_code, token, username)
            
            flash(f'Account created! A verification link has been sent to {email}.', 'success')
            return redirect(url_for('auth.login'))
        else:
            flash('Username or email already exists', 'error')
    
    return render_template('register.html')

@auth_bp.route('/verify/<token>')
def verify_email(token):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE verification_token = ? AND email_verified = 0', (token,)).fetchone()
    
    if not user:
        conn.close()
        flash('Invalid or expired verification link.', 'error')
        return redirect(url_for('auth.login'))
    
    expires = datetime.fromisoformat(user['verification_token_expires']) if user['verification_token_expires'] else None
    if expires and expires < datetime.now():
        conn.close()
        flash('Verification link has expired. Please register again.', 'error')
        return redirect(url_for('auth.register'))
    
    conn.execute('UPDATE users SET email_verified = 1, verification_token = NULL, verification_token_expires = NULL WHERE id = ?', (user['id'],))
    conn.commit()
    conn.close()
    
    flash('✅ Email verified successfully! You can now log in.', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out', 'success')
    return redirect(url_for('auth.login'))

# ============================================================================
# PASSWORD RESET ROUTES
# ============================================================================

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(url_for('video.index'))
    
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
                return redirect(url_for('auth.reset_with_code'))
            else:
                flash('If this email exists, you will receive a code.', 'info')
                return redirect(url_for('auth.login'))
    
    return render_template('forgot_password.html')

@auth_bp.route('/reset-with-code', methods=['GET', 'POST'])
def reset_with_code():
    if 'reset_user_id' not in session:
        flash('Session expired. Please try again.', 'error')
        return redirect(url_for('auth.forgot_password'))
    
    user_id = session['reset_user_id']
    user = get_user_by_id(user_id)
    
    if not user:
        flash('User not found. Please try again.', 'error')
        session.pop('reset_user_id', None)
        return redirect(url_for('auth.forgot_password'))
    
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
                return redirect(url_for('auth.forgot_password'))
            
            is_valid, error_msg = validate_password(new_password, confirm_password)
            
            if not is_valid:
                flash(error_msg, 'error')
                return render_template('reset_with_code.html', step='password', email=user['email'])
            
            if update_user_password(user_id, new_password):
                updated_user = get_user_by_id(user_id)
                new_permanent_code = updated_user['reset_code']
                send_new_code_email(user['email'], new_permanent_code, user['username'])
                
                session.pop('reset_user_id', None)
                session.pop('code_verified', None)
                
                flash('Password reset! A new permanent code has been sent to your email.', 'success')
                return redirect(url_for('auth.login'))
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