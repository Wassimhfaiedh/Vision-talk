"""
Email utilities for Vision-Talk.
Handles email sending and HTML templates.
"""

import secrets
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import render_template, url_for
from config import Config

config = Config()


def generate_reset_code() -> str:
    """Generate a 6-digit reset code."""
    return ''.join(secrets.choice(string.digits) for _ in range(6))


def generate_verification_token() -> str:
    """Generate a secure verification token."""
    return secrets.token_urlsafe(32)


def get_registration_email_html(username: str, code: str, verification_url: str, email: str) -> str:
    return render_template('email_templates.html', 
                          template_name='registration_with_verification',
                          username=username, code=code,
                          verification_url=verification_url, email=email)


def get_email_verification_html(username: str, verification_url: str, new_email: str) -> str:
    return render_template('email_templates.html',
                          template_name='email_verification',
                          username=username, verification_url=verification_url, new_email=new_email)


def get_reset_code_email_html(username: str, code: str, email: str) -> str:
    return render_template('email_templates.html',
                          template_name='reset_code',
                          username=username, code=code, email=email)


def get_new_code_email_html(username: str, code: str, email: str) -> str:
    return render_template('email_templates.html',
                          template_name='new_code',
                          username=username, code=code, email=email)


def send_html_email(to_email: str, subject: str, html_content: str, text_content: str = None) -> bool:
    """Send HTML email using SMTP."""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = config.SMTP_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        if text_content:
            msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
        server.starttls()
        server.login(config.SMTP_EMAIL, config.SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False


def send_registration_email(to_email: str, code: str, verification_token: str, username: str):
    """Send registration verification email."""
    verification_url = url_for('auth.verify_email', token=verification_token, _external=True)
    subject = f"🔐 Vision-Talk - Verify Your Email"
    html_content = get_registration_email_html(username, code, verification_url, to_email)
    text_content = f"Hello {username},\n\nClick to verify: {verification_url}\n\nYour recovery code: {code}"
    send_html_email(to_email, subject, html_content, text_content)


def send_email_change_verification(to_email: str, verification_token: str, username: str):
    """Send email change verification email."""
    verification_url = url_for('profile.verify_new_email', token=verification_token, _external=True)
    subject = f"🔐 Vision-Talk - Verify Your New Email"
    html_content = get_email_verification_html(username, verification_url, to_email)
    text_content = f"Hello {username},\n\nClick to verify your new email: {verification_url}"
    send_html_email(to_email, subject, html_content, text_content)


def send_reset_code_email(to_email: str, code: str, username: str):
    """Send password reset code email."""
    subject = f"🔐 Vision-Talk - Password Reset Code"
    html_content = get_reset_code_email_html(username, code, to_email)
    text_content = f"Hello {username},\n\nYour temporary reset code is: {code}\nExpires in 10 minutes."
    send_html_email(to_email, subject, html_content, text_content)


def send_new_code_email(to_email: str, code: str, username: str):
    """Send new permanent recovery code email."""
    subject = f"🔐 Vision-Talk - Your New Recovery Code"
    html_content = get_new_code_email_html(username, code, to_email)
    text_content = f"Hello {username},\n\nYour NEW permanent recovery code is: {code}"
    send_html_email(to_email, subject, html_content, text_content)