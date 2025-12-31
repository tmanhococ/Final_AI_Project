"""
Rate Limiter Utility for Gemini API.

Provides shared rate limiting for all LLM calls in the evaluation module
to avoid hitting Gemini free tier limits (5 requests/minute).

Author: AI Evaluation Framework
"""

from __future__ import annotations

import time
import threading
from typing import Optional


class RateLimiter:
    """
    Thread-safe rate limiter for API calls.
    
    Implements a simple delay-based rate limiting strategy
    suitable for Gemini free tier (5 requests/minute = 12s delay).
    """
    
    # Singleton instance
    _instance: Optional["RateLimiter"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, delay_seconds: float = 13.0):
        """
        Initialize rate limiter.
        
        Args:
            delay_seconds: Minimum delay between API calls (default: 13s for safety with 5 req/min limit)
        """
        if self._initialized:
            return
        
        self.delay_seconds = delay_seconds
        self._last_call_time = 0.0
        self._call_lock = threading.Lock()
        self._initialized = True
    
    def wait(self, verbose: bool = True) -> float:
        """
        Wait if necessary to respect rate limits.
        
        Args:
            verbose: Print wait message
            
        Returns:
            Actual wait time in seconds
        """
        with self._call_lock:
            elapsed = time.time() - self._last_call_time
            
            if elapsed < self.delay_seconds:
                wait_time = self.delay_seconds - elapsed
                if verbose:
                    print(f"  ⏳ Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                wait_time = 0.0
            
            self._last_call_time = time.time()
            return wait_time
    
    def reset(self):
        """Reset the last call time."""
        with self._call_lock:
            self._last_call_time = 0.0
    
    @classmethod
    def set_delay(cls, delay_seconds: float):
        """
        Set the delay for all future waits.
        
        Args:
            delay_seconds: New delay in seconds
        """
        instance = cls()
        instance.delay_seconds = delay_seconds


# Global rate limiter instance with 45s delay (safe for free tier daily limits)
RATE_LIMITER = RateLimiter(delay_seconds=45.0)


def wait_for_rate_limit(verbose: bool = True) -> float:
    """
    Convenience function to wait for rate limit.
    
    Usage:
        from src.chatbot.evaluation.rate_limiter import wait_for_rate_limit
        
        wait_for_rate_limit()
        response = llm.invoke(prompt)
    
    Args:
        verbose: Print wait message
        
    Returns:
        Actual wait time in seconds
    """
    return RATE_LIMITER.wait(verbose=verbose)


def set_rate_limit_delay(delay_seconds: float):
    """
    Set the rate limit delay globally.
    
    Args:
        delay_seconds: Delay between API calls
    """
    RATE_LIMITER.delay_seconds = delay_seconds
