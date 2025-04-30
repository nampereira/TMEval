import time
import random
from typing import Dict, Any, Callable, Optional, TypeVar, cast
from datetime import datetime, timedelta
import threading

T = TypeVar('T')  # Generic type for return value of wrapped function

class ThrottlingManager:
    """Manages API call throttling to respect rate limits."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with throttling configuration.
        
        Args:
            config: Throttling configuration including requests_per_minute, 
                   retry_attempts, and backoff_factor
        """
        # Get global throttling settings
        global_throttling = config.get('throttling', {})
        self.enabled = global_throttling.get('enabled', True)
        self.requests_per_minute = global_throttling.get('requests_per_minute', 20)
        self.retry_attempts = global_throttling.get('retry_attempts', 3)
        self.backoff_factor = global_throttling.get('backoff_factor', 2.0)
        
        # Initialize tracking variables
        self.last_request_time = datetime.now()
        self.request_times = []
        self.lock = threading.Lock()  # For thread safety
    
    def with_throttling(self, 
                        func: Callable[..., T], 
                        llm_name: Optional[str] = None,
                        llm_config: Optional[Dict[str, Any]] = None) -> Callable[..., T]:
        """
        Decorator to apply throttling to a function.
        
        Args:
            func: Function to wrap with throttling
            llm_name: Name of the LLM for logging
            llm_config: LLM-specific configuration that may include throttling settings
            
        Returns:
            Wrapped function with throttling applied
        """
        # Adjust settings if LLM-specific config is provided
        requests_per_minute = self.requests_per_minute
        retry_attempts = self.retry_attempts
        backoff_factor = self.backoff_factor
        
        if llm_config and 'throttling' in llm_config:
            llm_throttling = llm_config['throttling']
            requests_per_minute = llm_throttling.get('requests_per_minute', requests_per_minute)
            retry_attempts = llm_throttling.get('retry_attempts', retry_attempts)
            backoff_factor = llm_throttling.get('backoff_factor', backoff_factor)
        
        def wrapped(*args: Any, **kwargs: Any) -> T:
            """Wrapped function with throttling."""
            if not self.enabled:
                return func(*args, **kwargs)
            
            # Calculate minimum time between requests
            min_interval = 60.0 / requests_per_minute  # in seconds
            
            for attempt in range(retry_attempts + 1):
                # Wait if needed to respect rate limits
                with self.lock:
                    # Clean up old request times (older than 1 minute)
                    now = datetime.now()
                    self.request_times = [t for t in self.request_times 
                                         if now - t < timedelta(minutes=1)]
                    
                    # Check if we're over the rate limit
                    if len(self.request_times) >= requests_per_minute:
                        # Calculate time to wait
                        oldest_time = min(self.request_times)
                        wait_time = (oldest_time + timedelta(minutes=1) - now).total_seconds()
                        if wait_time > 0:
                            llm_info = f" for {llm_name}" if llm_name else ""
                            print(f"Rate limit reached{llm_info}. Waiting {wait_time:.2f} seconds...")
                            time.sleep(wait_time + 0.1)  # Add a small buffer
                    
                    # Add jitter to avoid thundering herd problem when multiple threads hit limits
                    jitter = random.uniform(0, 0.5)
                    time_since_last = (now - self.last_request_time).total_seconds()
                    
                    if time_since_last < min_interval:
                        wait_time = min_interval - time_since_last + jitter
                        time.sleep(wait_time)
                    
                    # Record this request
                    self.last_request_time = datetime.now()
                    self.request_times.append(self.last_request_time)
                
                try:
                    # Call the actual function
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this is the last attempt
                    if attempt >= retry_attempts:
                        raise
                    
                    # Calculate backoff time with jitter
                    backoff_time = (backoff_factor ** attempt) * (1 + random.uniform(0, 0.5))
                    llm_info = f" for {llm_name}" if llm_name else ""
                    error_type = type(e).__name__
                    
                    print(f"API call{llm_info} failed with {error_type}: {str(e)}")
                    print(f"Retrying in {backoff_time:.2f} seconds (attempt {attempt+1}/{retry_attempts})...")
                    time.sleep(backoff_time)
            
            # This should never be reached because the last attempt either returns or raises
            return cast(T, None)
            
        return wrapped
