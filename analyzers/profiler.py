#!/usr/bin/env python3
"""
Python profiler module for performance analysis.

This module provides utilities for profiling Python functions and formatting
profiling results for analysis and LLM consumption.
"""

import cProfile
import pstats
import time
from dataclasses import dataclass
from typing import Callable, Any, List, Optional
from io import StringIO


@dataclass
class FunctionStats:
    """Statistics for a single function call."""
    name: str
    ncalls: int
    tottime: float  # Total time in this function (excluding subcalls)
    cumtime: float  # Cumulative time (including subcalls)
    percall_tot: float  # tottime / ncalls
    percall_cum: float  # cumtime / ncalls


@dataclass
class ProfileSummary:
    """Summary of profiling results."""
    total_time: float
    function_stats: List[FunctionStats]
    top_functions: List[FunctionStats]  # Top N by total time
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Total execution time: {self.total_time:.4f} seconds",
            f"\nTop {len(self.top_functions)} functions by time:",
            "-" * 80,
        ]
        
        for stat in self.top_functions:
            lines.append(
                f"{stat.name:50s} {stat.ncalls:8d} calls "
                f"{stat.tottime:8.4f}s total {stat.cumtime:8.4f}s cumulative"
            )
        
        return "\n".join(lines)


def profile_function(
    func: Callable,
    *args,
    n_runs: int = 1,
    **kwargs
) -> ProfileSummary:
    """
    Profile a callable using cProfile.
    
    Args:
        func: The callable to profile
        *args: Positional arguments to pass to func
        n_runs: Number of times to run the function (default: 1)
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        ProfileSummary containing profiling statistics
    
    Example:
        >>> def slow_function(n):
        ...     return sum(i**2 for i in range(n))
        >>> summary = profile_function(slow_function, 10000, n_runs=5)
        >>> print(summary)
    """
    profiler = cProfile.Profile()
    
    # Run the function n_runs times under profiling
    profiler.enable()
    for _ in range(n_runs):
        func(*args, **kwargs)
    profiler.disable()
    
    # Extract statistics
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('tottime')
    
    # Get total time
    total_time = sum(stat[2] for stat in stats.stats.values())
    
    # Extract function statistics
    function_stats = []
    for func_key, stat_data in stats.stats.items():
        # func_key is (filename, line_number, function_name)
        filename, line_num, func_name = func_key
        ncalls, tottime, cumtime = stat_data[0], stat_data[2], stat_data[3]
        
        # Calculate per-call times
        percall_tot = tottime / ncalls if ncalls > 0 else 0
        percall_cum = cumtime / ncalls if ncalls > 0 else 0
        
        function_stats.append(FunctionStats(
            name=f"{filename}:{line_num}({func_name})",
            ncalls=ncalls,
            tottime=tottime,
            cumtime=cumtime,
            percall_tot=percall_tot,
            percall_cum=percall_cum
        ))
    
    # Sort by total time (descending) and get top functions
    function_stats.sort(key=lambda x: x.tottime, reverse=True)
    top_functions = function_stats[:10]  # Top 10 by default
    
    return ProfileSummary(
        total_time=total_time,
        function_stats=function_stats,
        top_functions=top_functions
    )


def format_profile_summary(summary: ProfileSummary, llm_friendly: bool = False) -> str:
    """
    Format a ProfileSummary for display or LLM consumption.
    
    Args:
        summary: The ProfileSummary to format
        llm_friendly: If True, format for LLM prompt; otherwise human-readable
    
    Returns:
        Formatted string representation of the profile
    
    Example:
        >>> text = format_profile_summary(summary, llm_friendly=True)
        >>> # text can be included in LLM prompt
    """
    if llm_friendly:
        # Format for LLM consumption - more concise and structured
        lines = [
            "## Performance Profile Summary",
            f"Total execution time: {summary.total_time:.4f} seconds",
            "",
            "### Top Bottleneck Functions:",
            ""
        ]
        
        for i, stat in enumerate(summary.top_functions, 1):
            lines.append(
                f"{i}. {stat.name}\n"
                f"   - Called {stat.ncalls} times\n"
                f"   - Total time: {stat.tottime:.4f}s ({stat.tottime/summary.total_time*100:.1f}% of total)\n"
                f"   - Cumulative time: {stat.cumtime:.4f}s\n"
                f"   - Average per call: {stat.percall_tot:.6f}s"
            )
        
        lines.append("\nFocus optimization efforts on the functions with highest total time.")
        return "\n".join(lines)
    else:
        # Human-readable format with more details
        return str(summary)


def measure_execution_time(func: Callable, *args, n_runs: int = 10, **kwargs) -> float:
    """
    Measure the average execution time of a function using time.perf_counter.
    
    This is a simpler alternative to full profiling when you just need timing.
    
    Args:
        func: The callable to measure
        *args: Positional arguments to pass to func
        n_runs: Number of times to run for averaging (default: 10)
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Average execution time in seconds
    
    Example:
        >>> def my_function(n):
        ...     return sum(range(n))
        >>> avg_time = measure_execution_time(my_function, 10000, n_runs=100)
        >>> print(f"Average time: {avg_time:.6f}s")
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return sum(times) / len(times)
