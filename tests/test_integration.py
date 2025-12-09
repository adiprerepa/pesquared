#!/usr/bin/env python3
"""
Integration tests for the complete Python optimization pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.optimizer import optimize_with_llm
from optimizers.llm_client import create_llm_client
from verifiers.verifier import check_correctness, compare_performance


def test_end_to_end_optimization():
    """Test complete optimization workflow with dummy client."""
    
    # Define a simple function to optimize
    def simple_sum(n):
        """Sum numbers from 0 to n-1."""
        result = 0
        for i in range(n):
            result += i
        return result
    
    # Create dummy client
    client = create_llm_client('dummy')
    
    # Run optimization
    result = optimize_with_llm(simple_sum, 100, llm_client=client, n_runs=3)
    
    # Verify result structure
    assert result.original_code is not None
    assert result.optimized_code is not None
    assert result.explanation is not None
    assert result.model_used == 'dummy-model'
    assert result.profile_summary is not None
    
    # Verify the optimized code can be executed
    namespace = {}
    exec(result.optimized_code, namespace)
    
    # Find the function in namespace
    optimized_func = None
    for name, obj in namespace.items():
        if callable(obj) and not name.startswith('_'):
            optimized_func = obj
            break
    
    assert optimized_func is not None, "Should have found optimized function"
    
    # Verify correctness
    test_cases = [
        ((10,), {}),
        ((100,), {}),
        ((0,), {}),
    ]
    
    passed, errors = check_correctness(simple_sum, optimized_func, test_cases)
    assert passed, f"Correctness check failed: {errors}"
    
    print("✅ End-to-end optimization workflow successful")


def test_optimization_with_profiling():
    """Test that profiling data is included in optimization result."""
    
    def test_func(n):
        return sum(range(n))
    
    client = create_llm_client('dummy')
    result = optimize_with_llm(test_func, 100, llm_client=client, n_runs=5)
    
    # Verify profiling data
    assert result.profile_summary.total_time > 0
    assert len(result.profile_summary.function_stats) > 0
    assert len(result.profile_summary.top_functions) > 0
    
    print("✅ Profiling data correctly captured")


def test_performance_comparison_workflow():
    """Test complete performance comparison workflow."""
    
    def original(n):
        result = []
        for i in range(n):
            result.append(i * 2)
        return result
    
    def optimized(n):
        return [i * 2 for i in range(n)]
    
    # Compare performance
    comparison = compare_performance(original, optimized, 1000, n_runs=10)
    
    # Verify comparison structure
    assert 'original_time' in comparison
    assert 'optimized_time' in comparison
    assert 'speedup' in comparison
    assert 'improvement_pct' in comparison
    
    assert comparison['original_time'] > 0
    assert comparison['optimized_time'] > 0
    
    print("✅ Performance comparison workflow successful")


if __name__ == '__main__':
    # Simple test runner
    import traceback
    
    tests = [
        test_end_to_end_optimization,
        test_optimization_with_profiling,
        test_performance_comparison_workflow,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
