import cProfile
import functools
import inspect
import io
import linecache
import os
import pstats
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Union

class Profiler:
    """
    A comprehensive profiling utility for Python functions and code blocks.
    
    Features:
    - Function and code block time profiling
    - Memory profiling (optional)
    - Decorator for easy function profiling
    - Context manager for block profiling
    - Customizable output and reporting
    """

    def __init__(self, output_dir: str = 'profile_results', 
                 enable_memory_profile: bool = False):
        """
        Initialize the Profiler.
        
        :param output_dir: Directory to save profiling results
        :param enable_memory_profile: Enable memory profiling (requires memory_profiler)
        """
        self.output_dir = output_dir
        self.enable_memory_profile = enable_memory_profile
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Optional memory profiler import
        if enable_memory_profile:
            try:
                import memory_profiler
            except ImportError:
                print("Memory profiler not installed. Install with 'pip install memory_profiler'")
                self.enable_memory_profile = False

    def profile_func(self, 
                     func: Callable, 
                     sort_by: str = 'cumulative', 
                     lines: int = 50) -> Any:
        """
        Profile a function using cProfile.
        
        :param func: Function to profile
        :param sort_by: Sort results by this metric
        :param lines: Number of lines to print
        :return: Function result
        """
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats(sort_by)
            
            # Generate output filename
            output_file = os.path.join(
                self.output_dir, 
                f'profile_{func.__name__}_{int(time.time())}.txt'
            )
            
            # Redirect stdout to file
            with open(output_file, 'w') as f:
                sys.stdout = f
                stats.print_stats(lines)
                sys.stdout = sys.__stdout__
            
            print(f"Profiling results for {func.__name__} saved to {output_file}")

    def profile_decorator(self, 
                          sort_by: str = 'cumulative', 
                          lines: int = 50):
        """
        Decorator to profile a function.
        
        :param sort_by: Sort results by this metric
        :param lines: Number of lines to print
        :return: Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.profile_func(
                    lambda: func(*args, **kwargs), 
                    sort_by, 
                    lines
                )
            return wrapper
        return decorator

    @contextmanager
    def profile_block(self, 
                      block_name: str = 'code_block', 
                      sort_by: str = 'cumulative', 
                      lines: int = 50):
        """
        Context manager for profiling a code block.
        
        :param block_name: Name of the code block
        :param sort_by: Sort results by this metric
        :param lines: Number of lines to print
        """
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats(sort_by)
            
            # Generate output filename
            output_file = os.path.join(
                self.output_dir, 
                f'profile_block_{block_name}_{int(time.time())}.txt'
            )
            
            # Redirect stdout to file
            with open(output_file, 'w') as f:
                sys.stdout = f
                stats.print_stats(lines)
                sys.stdout = sys.__stdout__
            
            print(f"Block profiling results saved to {output_file}")

    def memory_profile_func(self, func: Callable) -> Any:
        """
        Memory profile a function.
        
        :param func: Function to profile
        :return: Function result
        """
        if not self.enable_memory_profile:
            raise ImportError("Memory profiler not enabled. Install memory_profiler.")
        
        from memory_profiler import memory_usage
        
        # Capture memory usage and function result
        mem_usage, result = memory_usage((func, [], {}), retval=True, max_iterations=1)
        
        # Generate output filename
        output_file = os.path.join(
            self.output_dir, 
            f'memory_profile_{func.__name__}_{int(time.time())}.txt'
        )
        
        # Write memory usage results
        with open(output_file, 'w') as f:
            f.write(f"Memory Usage for {func.__name__}:\n")
            f.write(f"Peak Memory: {max(mem_usage)} MiB\n")
            f.write(f"Memory Usage Trace: {mem_usage}\n")
        
        print(f"Memory profiling results for {func.__name__} saved to {output_file}")
        return result

    def print_line_by_line_profile(self, 
                                   func: Callable, 
                                   output_file: Optional[str] = None):
        """
        Generate a line-by-line profile of a function.
        
        :param func: Function to profile
        :param output_file: Optional file to save results
        """
        import inspect
        
        # Get source code lines
        lines, start_line = inspect.getsourcelines(func)
        
        # Profile the function
        profiler = cProfile.Profile()
        profiler.enable()
        func()  # Run the function
        profiler.disable()
        
        # Prepare output
        output = []
        
        # Process line-by-line stats
        for i, line in enumerate(lines, start=start_line):
            line = line.strip()
            # Find line hits in profiler stats
            hits = sum(1 for entry in profiler.getstats() if entry.lineno == start_line + i)
            output.append(f"{start_line + i}: {hits} hits - {line}")
        
        # Determine output destination
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(output))
        else:
            print('\n'.join(output))

# Example usage demonstrating all profiler features
def example_usage():
    # Create profiler instance
    profiler = Profiler(output_dir='./profiler_results', enable_memory_profile=True)
    
    # Example function to profile
    def sample_function(n):
        return sum(i**2 for i in range(n))
    
    # Using function profiler
    profiler.profile_func(lambda: sample_function(10000))
    
    # Using decorator
    @profiler.profile_decorator()
    def decorated_function():
        return sample_function(20000)
    
    decorated_function()
    
    # Using context manager
    with profiler.profile_block('computation_block'):
        result = sample_function(30000)
    
    # Memory profiling
    try:
        profiler.memory_profile_func(lambda: sample_function(50000))
    except ImportError as e:
        print(f"Memory profiling skipped: {e}")

if __name__ == '__main__':
    example_usage()