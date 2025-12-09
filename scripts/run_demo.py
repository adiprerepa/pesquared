#!/usr/bin/env python3
"""
CLI demo script for PESquared Python optimization.

This script provides command-line access to various demo workflows without
requiring Jupyter notebooks.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzers.profiler import profile_function, format_profile_summary
from optimizers.optimizer import optimize_with_llm, construct_optimization_prompt
from optimizers.llm_client import create_llm_client
from verifiers.verifier import (
    check_correctness,
    compare_performance,
    print_performance_comparison
)


def demo_baseline():
    """Run the baseline profiling demo."""
    print("=" * 70)
    print("BASELINE PROFILING DEMO")
    print("=" * 70)
    print("\n1. Defining inefficient Fibonacci function...\n")
    
    def naive_fibonacci(n):
        """Inefficient recursive Fibonacci implementation."""
        if n <= 1:
            return n
        return naive_fibonacci(n - 1) + naive_fibonacci(n - 2)
    
    print("def naive_fibonacci(n):")
    print("    if n <= 1:")
    print("        return n")
    print("    return naive_fibonacci(n - 1) + naive_fibonacci(n - 2)")
    
    print("\n2. Profiling naive_fibonacci(20)...\n")
    
    # Profile the naive implementation
    summary = profile_function(naive_fibonacci, 20, n_runs=1)
    print(format_profile_summary(summary, llm_friendly=False))
    
    print("\n3. Defining optimized version...\n")
    
    def optimized_fibonacci(n):
        """Efficient iterative Fibonacci implementation."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    print("def optimized_fibonacci(n):")
    print("    if n <= 1:")
    print("        return n")
    print("    a, b = 0, 1")
    print("    for _ in range(2, n + 1):")
    print("        a, b = b, a + b")
    print("    return b")
    
    print("\n4. Comparing performance...\n")
    
    # Compare performance
    comparison = compare_performance(
        naive_fibonacci,
        optimized_fibonacci,
        25,
        n_runs=10
    )
    
    print_performance_comparison(comparison)
    
    print("✅ Demo complete! Key takeaway: Algorithmic improvements provide huge gains.\n")


def demo_llm(use_real_llm=False, model="gpt-3.5-turbo"):
    """Run the LLM-guided optimization demo."""
    print("=" * 70)
    print("LLM-GUIDED OPTIMIZATION DEMO")
    print("=" * 70)
    
    # Configure LLM client
    if use_real_llm:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("\n⚠️  OPENAI_API_KEY not set. Falling back to dummy client.")
            llm_client = create_llm_client('dummy')
        else:
            llm_client = create_llm_client('openai', model=model)
            print(f"\n✅ Using OpenAI model: {llm_client.get_model_name()}")
    else:
        llm_client = create_llm_client('dummy')
        print(f"\nℹ️  Using {llm_client.get_model_name()} (offline mode)")
    
    print("\n1. Defining function to optimize...\n")
    
    def process_data(numbers):
        """Process a list of numbers with several inefficiencies."""
        # Multiple passes over the data
        positive = [x for x in numbers if x > 0]
        squared = [x**2 for x in positive]
        
        # Unnecessary intermediate list
        total = sum([x for x in squared])
        
        # Repeated lookups
        result = []
        for x in squared:
            if x < total / len(squared):
                result.append(x)
        
        return result
    
    print("def process_data(numbers):")
    print("    positive = [x for x in numbers if x > 0]")
    print("    squared = [x**2 for x in positive]")
    print("    total = sum([x for x in squared])")
    print("    result = []")
    print("    for x in squared:")
    print("        if x < total / len(squared):")
    print("            result.append(x)")
    print("    return result")
    
    test_input = list(range(-50, 51))
    
    print("\n2. Running LLM-guided optimization...\n")
    
    # Run optimization
    optimization_result = optimize_with_llm(
        process_data,
        test_input,
        llm_client=llm_client,
        n_runs=5
    )
    
    print("✅ Optimization complete!\n")
    print("=" * 70)
    print(f"Model used: {optimization_result.model_used}")
    print("=" * 70)
    
    print("\n3. LLM Explanation:\n")
    print(optimization_result.explanation)
    
    print("\n4. Optimized Code:\n")
    print(optimization_result.optimized_code)
    
    print("\n5. Attempting to execute and verify...\n")
    
    # Try to execute the optimized code
    namespace = {}
    try:
        exec(optimization_result.optimized_code, namespace)
        
        # Find the optimized function
        optimized_func = namespace.get('process_data') or namespace.get('process_data_optimized')
        
        if optimized_func is None:
            funcs = [v for v in namespace.values() if callable(v) and not v.__name__.startswith('_')]
            if funcs:
                optimized_func = funcs[0]
        
        if optimized_func:
            print(f"✅ Successfully loaded optimized function: {optimized_func.__name__}")
            
            # Check correctness
            test_cases = [
                ((list(range(-50, 51)),), {}),
                ((list(range(-10, 11)),), {}),
                (([1, 2, 3, 4, 5],), {}),
            ]
            
            passed, errors = check_correctness(process_data, optimized_func, test_cases)
            
            if passed:
                print(f"✅ Correctness verified ({len(test_cases)} test cases passed)")
                
                # Measure performance
                print("\n6. Measuring performance improvement...\n")
                
                perf_test_input = list(range(-5000, 5001))
                comparison = compare_performance(
                    process_data,
                    optimized_func,
                    perf_test_input,
                    n_runs=50
                )
                
                print_performance_comparison(comparison)
                
                if comparison['speedup'] > 1.1:
                    print("🎉 Significant performance improvement achieved!")
                elif comparison['speedup'] > 1.0:
                    print("✅ Modest performance improvement achieved.")
                else:
                    print("⚠️  No significant improvement (may happen with dummy client).")
            else:
                print("❌ Correctness verification failed:")
                for error in errors:
                    print(f"\n{error}")
        else:
            print("❌ Could not find optimized function in generated code.")
            print("    (This is expected with the dummy client)")
    
    except Exception as e:
        print(f"❌ Error executing optimized code: {e}")
        print("    (This is expected with the dummy client)")
    
    print("\n✅ Demo complete!\n")


def demo_prompt(function_path, function_name):
    """Generate and print an LLM-ready prompt for a function."""
    print("=" * 70)
    print("LLM PROMPT GENERATION")
    print("=" * 70)
    
    # Import the function
    sys.path.insert(0, os.path.dirname(os.path.abspath(function_path)))
    module_name = os.path.splitext(os.path.basename(function_path))[0]
    
    try:
        module = __import__(module_name)
        func = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        print(f"❌ Error importing function: {e}")
        return
    
    print(f"\n1. Profiling {function_name}...\n")
    
    # Profile with dummy arguments (you may need to adjust this)
    try:
        summary = profile_function(func, n_runs=5)
    except TypeError:
        print("⚠️  Function requires arguments. Skipping profiling.")
        print("    Provide test arguments in the code for better results.")
        return
    
    print(format_profile_summary(summary, llm_friendly=False))
    
    print("\n2. Generating LLM-ready prompt...\n")
    
    import inspect
    import textwrap
    
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)
    
    prompt = construct_optimization_prompt(source_code, summary, function_name)
    
    print("=" * 70)
    print(prompt)
    print("=" * 70)
    
    print("\n✅ Prompt generated! Copy the above to your LLM of choice.\n")


def main():
    parser = argparse.ArgumentParser(
        description='PESquared Python Optimization Demos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline profiling demo
  python run_demo.py --demo baseline
  
  # Run LLM optimization demo (offline, no API key needed)
  python run_demo.py --demo llm
  
  # Run LLM optimization with real OpenAI API
  python run_demo.py --demo llm --use-real-llm --model gpt-4
  
  # Generate an LLM prompt for a custom function
  python run_demo.py --demo prompt --function my_module.py --name my_function
        """
    )
    
    parser.add_argument(
        '--demo',
        choices=['baseline', 'llm', 'prompt'],
        required=True,
        help='Which demo to run'
    )
    
    parser.add_argument(
        '--use-real-llm',
        action='store_true',
        help='Use real OpenAI LLM instead of dummy client (requires OPENAI_API_KEY)'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-3.5-turbo',
        help='OpenAI model to use (default: gpt-3.5-turbo)'
    )
    
    parser.add_argument(
        '--function',
        help='Path to Python file containing function (for prompt demo)'
    )
    
    parser.add_argument(
        '--name',
        help='Name of function to analyze (for prompt demo)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.demo == 'prompt':
        if not args.function or not args.name:
            parser.error("--function and --name are required for 'prompt' demo")
    
    # Run the selected demo
    try:
        if args.demo == 'baseline':
            demo_baseline()
        elif args.demo == 'llm':
            demo_llm(use_real_llm=args.use_real_llm, model=args.model)
        elif args.demo == 'prompt':
            demo_prompt(args.function, args.name)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
