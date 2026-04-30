"""
API key validation utilities for Vision-Talk.
"""

import requests
import time
from typing import Tuple, Optional


# =========================
# CONFIG
# =========================
TIMEOUT = 10
MAX_RETRIES = 2
BACKOFF = 1.5


# =========================
# RETRY REQUEST HELPER
# =========================
def _request_with_retry(method: str, url: str, headers: dict = None, json: dict = None) -> requests.Response:
    """Make request with retry logic for network errors."""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                timeout=TIMEOUT
            )
            if response.status_code in [429, 500, 502, 503, 504]:
                time.sleep(BACKOFF ** attempt)
                continue
            
            return response
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF ** attempt)
            continue
    
    raise last_error


# =========================
# GEMINI VALIDATION
# =========================
def _validate_gemini(api_key: str) -> Tuple[bool, Optional[str]]:
    """Validate Gemini API key."""
    url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
    
    try:
        response = _request_with_retry("GET", url)
        
        if response.status_code == 200:
            return True, None
        elif response.status_code == 401:
            return False, "Invalid Gemini API key"
        else:
            return False, f"Gemini validation failed (HTTP {response.status_code})"
            
    except Exception as e:
        return False, f"Gemini error: {str(e)}"


# =========================
# MOONDREAM VALIDATION
# =========================
def _validate_moondream(api_key: str) -> Tuple[bool, Optional[str]]:
    """Validate Moondream API key."""
    url = "https://api.moondream.ai/v1/query"
    headers = {"X-Moondream-Auth": api_key}
    payload = {
        "image_url": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "question": "What is this?"
    }
    
    try:
        response = _request_with_retry("POST", url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return True, None
        elif response.status_code == 401:
            return False, "Invalid Moondream API key"
        else:
            return True, None
            
    except Exception as e:
        return False, f"Moondream error: {str(e)}"


# =========================
# NVIDIA VALIDATION 
# =========================
def _validate_nvidia(api_key: str) -> Tuple[bool, Optional[str]]:
    """Validate NVIDIA API key using chat completions endpoint."""
    # Check format first
    if not api_key.startswith('nvapi-'):
        return False, "Invalid NVIDIA API key format (should start with nvapi-)"
    
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "accept": "application/json",
        "content-type": "application/json"
    }
    # Minimal test payload 
    payload = {
        "model": "nvidia/nemotron-nano-12b-v2-vl",
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "max_tokens": 1,
        "temperature": 0
    }
    
    try:
        response = _request_with_retry("POST", url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return True, None
        if response.status_code == 401:
            return False, "Invalid NVIDIA API key (unauthorized)"
        if response.status_code == 403:
            return False, "NVIDIA API key lacks required permissions. Please enable 'Public API Endpoints' in your NVIDIA account."
        if response.status_code == 400:
            return True, None
        # Rate limited
        if response.status_code == 429:
            return False, "Rate limited - please try again later"
        
        # Other status codes
        return False, f"NVIDIA API key validation failed (HTTP {response.status_code})"
        
    except Exception as e:
        return False, f"NVIDIA error: {str(e)}"


# =========================
# MAIN VALIDATION FUNCTION
# =========================
def validate_api_key(model_name: str, api_key: str) -> Tuple[bool, Optional[str]]:
    """Validate API key for different AI models."""
    
    # Gemini models
    if "Gemini" in model_name:
        return _validate_gemini(api_key)
    
    # Moondream models
    elif "Moondream" in model_name:
        return _validate_moondream(api_key)
    
    # NVIDIA models (NeMoVision, Llama 90B, Llama 11B)
    elif "NeMoVision" in model_name or "Llama" in model_name:
        return _validate_nvidia(api_key)
    
    # Unknown model
    return False, f"Unknown model type: {model_name}"