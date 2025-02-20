#!/usr/bin/env python3

import argparse
import os
import traceback
from dotenv import load_dotenv
from analyze_stacks import StackAnalyzer
from extract_deps import extract_dependencies, DependencyExtractorConfig, format_analysis_output
from optimize_function import FunctionOptimizer, OptimizationResult
from performance_verifier import PerformanceVerifier
from hw1 import APEHW1

# Load environment variables from .env file
load_dotenv()

def optimize_hotspots(codebase_dir: str, stacks_dir: str, openai_key: str, perf_verifier: PerformanceVerifier, num_functions: int = 3, model="gpt-4o") -> None:
    """
    Optimize the top hotspot functions in a loop.
    
    Args:
        codebase_dir: Directory containing the C++ codebase
        stacks_dir: Directory containing the folded stack files
        openai_key: OpenAI API key for optimization
        num_functions: Number of top functions to optimize (default: 3)
    """

    # Initialize analyzers
    stack_analyzer = StackAnalyzer(stacks_dir)
    optimizer = FunctionOptimizer(openai_api_key=openai_key, model=model)
    config = DependencyExtractorConfig(
        include_function_locations=True,
        include_type_locations=True
    )
    
    # Get iterator for functions sorted by exclusive time
    functions_iter = stack_analyzer.iter_functions()
    optimized_count = 0
    
    print(f"\nStarting optimization loop for top {num_functions} hotspot functions...")
    
    while optimized_count < num_functions:
        # 1. Get next hottest function
        try:
            func = next(functions_iter)
        except StopIteration:
            print("\nNo more functions to analyze")
            break
            
        # Skip unknown functions
        if "[unknown]" in func.name or not func.name.strip():
            continue
            
        print(f"\nAnalyzing hotspot #{optimized_count + 1}:")
        print(f"Function: {func.name}")
        print(f"Exclusive time: {func.exclusive_time:,} samples")
        
        try:
            # 2. Extract function dependencies and context
            print("Extracting dependencies...")
            try:
                analysis = extract_dependencies(codebase_dir, func.name, config)
            except Exception as e:
                print(f"Error extracting dependencies: {str(e)}")
                print("Stack trace:")
                traceback.print_exc()
                continue
            
            if not analysis.functions:
                print(f"Could not find function {func.name} in codebase. Skipping...")
                
            # 3. Generate optimization using LLM
            print("Generating optimization...")
            try:
                result = optimizer.optimize_function(codebase_dir, func.name, analysis, optimized_count=optimized_count)
            except Exception as e:
                print(f"Error generating optimization: {str(e)}")
                print("Stack trace:")
                traceback.print_exc()
                continue
            print(f"Generated function: {result.optimized_function}")
            
            # 4 & 5. Create branch and apply optimization
            print("Applying optimization...")
            try:
                optimizer.apply_optimization(result, codebase_dir)
            except Exception as e:
                print(f"Error applying optimization: {str(e)}")
                print("Stack trace:")
                traceback.print_exc()
                continue
            
            print(f"\nSuccess! Created branch: {result.branch_name}")
            print(f"Modified file: {result.file_path}")
            
            # 6. Verify performance improvement and tests pass
            print("Verifying performance improvement...")
            cur_perf = perf_verifier.get_performance()
            new_perf = perf_verifier.get_performance(result.branch_name)
            print(f"Current performance: {cur_perf}")
            print(f"New performance: {new_perf}")
            if perf_verifier.compare_performance(cur_perf, new_perf):
                print("Performance improved!")
                optimized_count += 1
            
        except Exception as e:
            print(f"Unexpected error optimizing function: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            print("Continuing to next function...\n")
            continue

def main():
    parser = argparse.ArgumentParser(
        description='Optimize performance hotspots in a C++ codebase using LLM'
    )
    parser.add_argument('codebase_dir',
                       help='Directory containing the C++ codebase')
    parser.add_argument('stacks_dir',
                       help='Directory containing the folded stack files')
    parser.add_argument('--openai-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--num-functions',
                       type=int,
                       default=3,
                       help='Number of top functions to optimize (default: 3)')
    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug output')
    parser.add_argument('--model',
                       default='gpt-3.5-turbo',
                       help='OpenAI model to use for optimization (default: gpt-3.5-turbo)')
    
    args = parser.parse_args()
    
    # Get API key
    openai_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("Error: OpenAI API key required")
        print("Set OPENAI_API_KEY environment variable or use --openai-key")
        return
    
    # Ensure codebase_dir is absolute path
    codebase_dir = os.path.abspath(args.codebase_dir)
    pv = APEHW1(codebase_dir)
    
    try:
        optimize_hotspots(
            codebase_dir,
            args.stacks_dir,
            openai_key,
            pv,
            args.num_functions,
            model=args.model
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        if args.debug:
            print("\nStack trace:")
            traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 