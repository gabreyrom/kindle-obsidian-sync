"""Rate limiting and exponential backoff quota management for Google Gemini API.

Guarantees compliance with Google AI Studio Free Tier limits:
- Automatic token bucket / minimum interval throttler (guarantees >= 4.2 seconds between sequential calls).
- Jittered exponential backoff retry decorator for HTTP 429 / RESOURCE_EXHAUSTED errors.
"""

from __future__ import annotations

import functools
import logging
import random
import re
import time
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Default interval to stay under 15 RPM Free Tier limit (60s / 15 = 4.0s -> 4.2s buffer)
DEFAULT_MIN_INTERVAL_SECONDS = 4.2


def extract_retry_delay(error: Exception, default_delay: float = 5.0) -> float:
    """Extract recommended retry delay (in seconds) from Gemini 429 quota errors if present."""
    err_str = str(error)
    match = re.search(
        r"retry(?:\s+in|\s*delay[\'\"]?\s*:\s*[\'\"]?)\s*(\d+(?:\.\d+)?)s?",
        err_str,
        re.IGNORECASE,
    )
    if match:
        try:
            delay = float(match.group(1)) + 1.0  # 1s safety buffer
            return max(delay, 2.0)
        except (ValueError, IndexError):
            pass
    return default_delay


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an exception is due to HTTP 429 / RESOURCE_EXHAUSTED / quota limits."""
    err_str = str(error).lower()
    return any(marker in err_str for marker in ("429", "resource_exhausted", "quota exceeded", "rate limit"))


def is_decommissioned_model_error(error: Exception) -> bool:
    """Check if model returned 404 / no longer available."""
    err_str = str(error).lower()
    return "404" in err_str or "not_found" in err_str or "no longer available" in err_str


class RateLimiter:
    """Token bucket / minimum interval throttler to guarantee Free Tier compliance."""

    def __init__(self, min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS):
        self.min_interval = float(min_interval_seconds)
        self._last_call_time: float = 0.0

    def acquire(self, on_cooldown: Optional[Callable[[float], None]] = None) -> None:
        """Throttle calls to guarantee at least `min_interval` seconds between sequential calls."""
        if self.min_interval <= 0:
            return

        now = time.time()
        if self._last_call_time > 0:
            elapsed = now - self._last_call_time
            if elapsed < self.min_interval:
                cooldown = self.min_interval - elapsed
                logger.info("RateLimiter: pacing call, waiting %.2fs...", cooldown)
                if on_cooldown:
                    on_cooldown(cooldown)
                time.sleep(cooldown)

        self._last_call_time = time.time()

    def record_call(self) -> None:
        """Record timestamp of completed or in-progress call."""
        self._last_call_time = time.time()


def retry_with_exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    on_retry: Optional[Callable[[int, float, Exception], None]] = None,
):
    """Decorator wrapping API calls with jittered exponential backoff for HTTP 429 errors.

    Formula: wait_time = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_err: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    # If decommissioned (404) or non-rate-limit, or max retries reached: raise immediately
                    if is_decommissioned_model_error(e) or not is_rate_limit_error(e) or attempt >= max_retries:
                        raise e

                    explicit_delay = extract_retry_delay(e, default_delay=0.0)
                    jitter = random.uniform(0.5, 1.5)
                    calc_delay = base_delay * (2 ** attempt) + jitter
                    wait_time = max(calc_delay, explicit_delay)

                    logger.warning(
                        "Encountered 429 RESOURCE_EXHAUSTED. Retrying in %.2fs (attempt %d/%d)...",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    if on_retry:
                        on_retry(attempt + 1, wait_time, e)

                    time.sleep(wait_time)

            if last_err:
                raise last_err
        return wrapper  # type: ignore[return-value]
    return decorator
