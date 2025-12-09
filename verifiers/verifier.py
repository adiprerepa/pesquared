#!/usr/bin/env python3
"""
Verification utilities for Python code optimization.

This module provides functions to measure runtime performance and verify
correctness of optimized code against the original implementation.
"""

import time
from typing import Callable, Any, List, Tuple, Optional


def measure_runtime(
    func: Callable,
    *args,
    n_runs: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    Measure the average and standard deviation of function runtime.
    
    Args:
        func: The function to measure
        *args: Positional arguments to pass to func
        n_runs: Number of runs for averaging (default: 100)
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Tuple of (average_time, std_deviation) in seconds
    
    Example:
        >>> def my_func(n):
        ...     return sum(range(n))
        >>> avg, std = measure_runtime(my_func, 10000, n_runs=100)
        >>> print(f"Average: {avg:.6f}s ± {std:.6f}s")
    """
    times = []
    
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    
    # Calculate standard deviation
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5
    
    return avg_time, std_dev


def check_correctness(
    original_func: Callable,
    optimized_func: Callable,
    test_cases: List[Tuple[tuple, dict]]
) -> Tuple[bool, List[str]]:
    """
    Verify that optimized function produces the same results as the original.
    
    Args:
        original_func: The original function
        optimized_func: The optimized function to verify
        test_cases: List of (args, kwargs) tuples to test
    
    Returns:
        Tuple of (all_passed, error_messages)
        - all_passed: True if all test cases passed
        - error_messages: List of error descriptions (empty if all passed)
    
    Example:
        >>> def original(x): return x * 2
        >>> def optimized(x): return x + x
        >>> test_cases = [((5,), {}), ((10,), {}), ((0,), {})]
        >>> passed, errors = check_correctness(original, optimized, test_cases)
        >>> assert passed
    """
    errors = []
    
    for i, (args, kwargs) in enumerate(test_cases):
        try:
            original_result = original_func(*args, **kwargs)
            optimized_result = optimized_func(*args, **kwargs)
            
            # Compare results (handles various types)
            if not _compare_results(original_result, optimized_result):
                errors.append(
                    f"Test case {i}: Results differ\n"
                    f"  Input: args={args}, kwargs={kwargs}\n"
                    f"  Original: {original_result}\n"
                    f"  Optimized: {optimized_result}"
                )
        except Exception as e:
            errors.append(
                f"Test case {i}: Exception in optimized function\n"
                f"  Input: args={args}, kwargs={kwargs}\n"
                f"  Error: {type(e).__name__}: {e}"
            )
    
    return len(errors) == 0, errors


def _compare_results(result1: Any, result2: Any, tolerance: float = 1e-9) -> bool:
    """
    Compare two results, handling different types appropriately.
    
    Args:
        result1: First result
        result2: Second result
        tolerance: Tolerance for floating point comparisons
    
    Returns:
        True if results are equivalent
    """
    # Handle None
    if result1 is None and result2 is None:
        return True
    if result1 is None or result2 is None:
        return False
    
    # Handle numeric types with tolerance
    if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
        return abs(result1 - result2) < tolerance
    
    # Handle lists/tuples recursively
    if isinstance(result1, (list, tuple)) and isinstance(result2, (list, tuple)):
        if len(result1) != len(result2):
            return False
        return all(_compare_results(r1, r2, tolerance) for r1, r2 in zip(result1, result2))
    
    # Handle dictionaries
    if isinstance(result1, dict) and isinstance(result2, dict):
        if set(result1.keys()) != set(result2.keys()):
            return False
        return all(_compare_results(result1[k], result2[k], tolerance) for k in result1)
    
    # Default: use equality
    return result1 == result2


def compare_performance(
    original_func: Callable,
    optimized_func: Callable,
    *args,
    n_runs: int = 100,
    **kwargs
) -> dict:
    """
    Compare performance of original vs optimized function.
    
    Args:
        original_func: The original function
        optimized_func: The optimized function
        *args: Arguments to pass to both functions
        n_runs: Number of runs for averaging
        **kwargs: Keyword arguments to pass to both functions
    
    Returns:
        Dictionary with performance comparison results:
        - original_time: Average time for original
        - optimized_time: Average time for optimized
        - speedup: Speedup factor (original_time / optimized_time)
        - improvement_pct: Percentage improvement
    
    Example:
        >>> def original(n): return sum([i**2 for i in range(n)])
        >>> def optimized(n): return sum(i**2 for i in range(n))
        >>> result = compare_performance(original, optimized, 10000, n_runs=50)
        >>> print(f"Speedup: {result['speedup']:.2f}x")
    """
    orig_time, orig_std = measure_runtime(original_func, *args, n_runs=n_runs, **kwargs)
    opt_time, opt_std = measure_runtime(optimized_func, *args, n_runs=n_runs, **kwargs)
    
    speedup = orig_time / opt_time if opt_time > 0 else float('inf')
    improvement_pct = ((orig_time - opt_time) / orig_time * 100) if orig_time > 0 else 0
    
    return {
        'original_time': orig_time,
        'original_std': orig_std,
        'optimized_time': opt_time,
        'optimized_std': opt_std,
        'speedup': speedup,
        'improvement_pct': improvement_pct
    }


def print_performance_comparison(comparison: dict) -> None:
    """
    Pretty-print a performance comparison.
    
    Args:
        comparison: Result from compare_performance()
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"Original:  {comparison['original_time']*1000:8.4f}ms ± {comparison['original_std']*1000:.4f}ms")
    print(f"Optimized: {comparison['optimized_time']*1000:8.4f}ms ± {comparison['optimized_std']*1000:.4f}ms")
    print("-" * 60)
    print(f"Speedup:   {comparison['speedup']:8.2f}x")
    print(f"Improvement: {comparison['improvement_pct']:6.1f}%")
    print("=" * 60 + "\n")
