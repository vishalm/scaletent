"""
Profiler utility for measuring execution time and memory usage in ScaleTent.
Provides both decorator and context manager interfaces for flexible profiling.
"""

import time
import cProfile
import pstats
import io
import functools
import tracemalloc
from typing import Optional, Callable, Any, Union
from contextlib import contextmanager
from loguru import logger

class TimeProfiler:
    """Time profiling utility that can be used as a decorator or context manager."""
    
    def __init__(self, name: str = None, detailed: bool = False):
        """
        Initialize the time profiler.
        
        Args:
            name (str, optional): Name for the profiling section
            detailed (bool): If True, uses cProfile for detailed profiling
        """
        self.name = name or "Code block"
        self.detailed = detailed
        self.profiler = cProfile.Profile() if detailed else None
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for the profiler."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
    
    def __enter__(self):
        """Context manager entry."""
        self.start_time = time.perf_counter()
        if self.detailed:
            self.profiler.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.detailed:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            logger.info(f"\n{self.name} Detailed Profile:\n{s.getvalue()}")
        
        elapsed = time.perf_counter() - self.start_time
        logger.info(f"{self.name} took {elapsed:.4f} seconds")

class MemoryProfiler:
    """Memory profiling utility that can be used as a decorator or context manager."""
    
    def __init__(self, name: str = None):
        """
        Initialize the memory profiler.
        
        Args:
            name (str, optional): Name for the profiling section
        """
        self.name = name or "Code block"
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for the profiler."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
    
    def __enter__(self):
        """Context manager entry."""
        tracemalloc.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"{self.name} Memory Usage:")
        logger.info(f"Current: {current / 1024:.2f} KB")
        logger.info(f"Peak: {peak / 1024:.2f} KB")

def profile(
    name: str = None,
    detailed: bool = False,
    profile_memory: bool = False
) -> Callable:
    """
    Combined profiler decorator for both time and memory profiling.
    
    Args:
        name (str, optional): Name for the profiling section
        detailed (bool): If True, uses cProfile for detailed time profiling
        profile_memory (bool): If True, enables memory profiling
    
    Returns:
        Callable: Decorated function with profiling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler_name = name or func.__name__
            with TimeProfiler(profiler_name, detailed=detailed):
                if profile_memory:
                    with MemoryProfiler(profiler_name):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage:
"""
# As a decorator
@profile(name="my_function", detailed=True, profile_memory=True)
def my_function():
    pass

# As context managers
with TimeProfiler("Time sensitive block"):
    # Your code here
    pass

with MemoryProfiler("Memory intensive block"):
    # Your code here
    pass
"""
