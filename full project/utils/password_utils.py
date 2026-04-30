"""
Password validation utilities for Vision-Talk.
"""

def validate_password(password: str, confirm_password: str = None) -> tuple:
    """Validate password strength."""
    if confirm_password is not None and password != confirm_password:
        return False, "Passwords do not match"
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if len(password) > 64:
        return False, "Password must be less than 64 characters"
    if ' ' in password:
        return False, "Password cannot contain spaces"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    if not (has_upper and has_lower and has_digit):
        return False, "Password must contain uppercase, lowercase, and numbers"
    
    return True, "OK"