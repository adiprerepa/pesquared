#!/usr/bin/env python3
"""
Tests for the verifier module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifiers.verifier import (
    measure_runtime,
    check_correctness,
    compare_performance,
    _compare_results
)


def test_measure_runtime():
    """Test runtime measurement."""
    def simple_func(n):
        return sum(range(n))
    
    avg_time, std_dev = measure_runtime(simple_func, 100, n_runs=10)
    
    assert avg_time > 0
    assert std_dev >= 0
    assert isinstance(avg_time, float)
    assert isinstance(std_dev, float)


def test_check_correctness_passing():
    """Test correctness check with matching functions."""
    def func1(x):
        return x * 2
    
    def func2(x):
        return x + x
    
    test_cases = [
        ((5,), {}),
        ((10,), {}),
        ((0,), {}),
    ]
    
    passed, errors = check_correctness(func1, func2, test_cases)
    
    assert passed is True
    assert len(errors) == 0


def test_check_correctness_failing():
    """Test correctness check with different functions."""
    def func1(x):
        return x * 2
    
    def func2(x):
        return x * 3  # Different result
    
    test_cases = [((5,), {})]
    
    passed, errors = check_correctness(func1, func2, test_cases)
    
    assert passed is False
    assert len(errors) > 0


def test_check_correctness_with_exception():
    """Test correctness check when optimized function raises exception."""
    def func1(x):
        return x * 2
    
    def func2(x):
        raise ValueError("Test error")
    
    test_cases = [((5,), {})]
    
    passed, errors = check_correctness(func1, func2, test_cases)
    
    assert passed is False
    assert len(errors) > 0
    assert "Exception" in errors[0]


def test_compare_results_numeric():
    """Test result comparison for numeric types."""
    assert _compare_results(5, 5) is True
    assert _compare_results(5.0, 5.0) is True
    assert _compare_results(5, 5.0) is True
    assert _compare_results(5, 6) is False


def test_compare_results_lists():
    """Test result comparison for lists."""
    assert _compare_results([1, 2, 3], [1, 2, 3]) is True
    assert _compare_results([1, 2, 3], [1, 2, 4]) is False
    assert _compare_results([1, 2], [1, 2, 3]) is False


def test_compare_results_dicts():
    """Test result comparison for dictionaries."""
    assert _compare_results({'a': 1, 'b': 2}, {'a': 1, 'b': 2}) is True
    assert _compare_results({'a': 1}, {'a': 2}) is False
    assert _compare_results({'a': 1}, {'b': 1}) is False


def test_compare_results_none():
    """Test result comparison for None values."""
    assert _compare_results(None, None) is True
    assert _compare_results(None, 5) is False
    assert _compare_results(5, None) is False


def test_compare_performance():
    """Test performance comparison."""
    def slow_func(n):
        result = 0
        for i in range(n):
            result += i
        return result
    
    def fast_func(n):
        return sum(range(n))
    
    comparison = compare_performance(slow_func, fast_func, 1000, n_runs=10)
    
    assert 'original_time' in comparison
    assert 'optimized_time' in comparison
    assert 'speedup' in comparison
    assert 'improvement_pct' in comparison
    assert comparison['original_time'] > 0
    assert comparison['optimized_time'] > 0


def test_check_correctness_with_kwargs():
    """Test correctness check with keyword arguments."""
    def func1(x, multiplier=2):
        return x * multiplier
    
    def func2(x, multiplier=2):
        return x * multiplier
    
    test_cases = [
        ((5,), {}),
        ((5,), {'multiplier': 3}),
    ]
    
    passed, errors = check_correctness(func1, func2, test_cases)
    
    assert passed is True
    assert len(errors) == 0


if __name__ == '__main__':
    # Simple test runner
    import traceback
    
    tests = [
        test_measure_runtime,
        test_check_correctness_passing,
        test_check_correctness_failing,
        test_check_correctness_with_exception,
        test_compare_results_numeric,
        test_compare_results_lists,
        test_compare_results_dicts,
        test_compare_results_none,
        test_compare_performance,
        test_check_correctness_with_kwargs,
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
