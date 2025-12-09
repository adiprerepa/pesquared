#!/usr/bin/env python3
"""
Tests for the profiler module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.profiler import (
    profile_function,
    format_profile_summary,
    measure_execution_time,
    ProfileSummary,
    FunctionStats
)


def test_profile_simple_function():
    """Test profiling a simple function."""
    def simple_sum(n):
        return sum(range(n))
    
    summary = profile_function(simple_sum, 100, n_runs=5)
    
    assert isinstance(summary, ProfileSummary)
    assert summary.total_time > 0
    assert len(summary.function_stats) > 0
    assert len(summary.top_functions) > 0


def test_profile_summary_str():
    """Test ProfileSummary string representation."""
    stats = [
        FunctionStats("test.py:1(func1)", 10, 0.5, 0.6, 0.05, 0.06),
        FunctionStats("test.py:2(func2)", 5, 0.3, 0.4, 0.06, 0.08),
    ]
    
    summary = ProfileSummary(
        total_time=1.0,
        function_stats=stats,
        top_functions=stats[:1]
    )
    
    str_repr = str(summary)
    assert "Total execution time: 1.0000 seconds" in str_repr
    assert "func1" in str_repr


def test_format_profile_summary_human_readable():
    """Test human-readable profile summary formatting."""
    stats = [FunctionStats("test.py:1(func)", 10, 0.5, 0.6, 0.05, 0.06)]
    summary = ProfileSummary(1.0, stats, stats)
    
    formatted = format_profile_summary(summary, llm_friendly=False)
    
    assert "Total execution time" in formatted
    assert "func" in formatted


def test_format_profile_summary_llm_friendly():
    """Test LLM-friendly profile summary formatting."""
    stats = [FunctionStats("test.py:1(func)", 10, 0.5, 0.6, 0.05, 0.06)]
    summary = ProfileSummary(1.0, stats, stats)
    
    formatted = format_profile_summary(summary, llm_friendly=True)
    
    assert "Performance Profile Summary" in formatted
    assert "Bottleneck Functions" in formatted
    assert "Focus optimization" in formatted


def test_measure_execution_time():
    """Test execution time measurement."""
    def simple_func(n):
        return sum(range(n))
    
    avg_time = measure_execution_time(simple_func, 1000, n_runs=10)
    
    assert avg_time > 0
    assert isinstance(avg_time, float)


def test_multiple_runs():
    """Test that multiple runs average correctly."""
    def constant_time_func():
        x = 1 + 1
        return x
    
    summary = profile_function(constant_time_func, n_runs=10)
    
    # Should have recorded some time
    assert summary.total_time > 0


if __name__ == '__main__':
    # Simple test runner
    import traceback
    
    tests = [
        test_profile_simple_function,
        test_profile_summary_str,
        test_format_profile_summary_human_readable,
        test_format_profile_summary_llm_friendly,
        test_measure_execution_time,
        test_multiple_runs,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
