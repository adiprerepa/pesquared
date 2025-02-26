#!/usr/bin/env python3

import argparse
import os
import traceback
import logging
from dotenv import load_dotenv
from analyze_stacks import StackAnalyzer
from extract_deps import extract_dependencies, DependencyExtractorConfig, format_analysis_output
from optimize_function import FunctionOptimizer, OptimizationResult
from performance_verifier import PerformanceVerifier
from hw1 import APEHW1

# Load environment variables from .env file
load_dotenv()

def setup_logger(debug: bool):
    """Set up logger with specified debug level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=level,
    )

def optimize_hotspots(codebase_dir: str, stacks_dir: str, api_key: str, perf_verifier: PerformanceVerifier, num_functions: int = 3, model="gpt-4o", provider="openai") -> None:
    """
    Optimize the top hotspot functions in a loop.
    
    Args:
        codebase_dir: Directory containing the C++ codebase
        stacks_dir: Directory containing the folded stack files
        openai_key: OpenAI API key for optimization
        num_functions: Number of top functions to optimize (default: 3)
    """

    logger = logging.getLogger()
    stack_analyzer = StackAnalyzer(stacks_dir)
    optimizer = FunctionOptimizer(api_key=api_key, model=model, provider=provider)
    config = DependencyExtractorConfig(
        include_function_locations=True,
        include_type_locations=True
    )
    
    functions_iter = stack_analyzer.iter_functions()
    optimized_count = 0
    
    logger.info(f"Starting optimization loop for top {num_functions} hotspot functions...")

    original_perf = perf_verifier.get_performance()
    
    while optimized_count < num_functions:
        try:
            func = next(functions_iter)
        except StopIteration:
            logger.info("No more functions to analyze")
            break
        
        if "[unknown]" in func.name or not func.name.strip():
            continue
        
        logger.info(f"\nAnalyzing hotspot #{optimized_count + 1}:")
        logger.info(f"Function: {func.name}")
        logger.info(f"Exclusive time: {func.exclusive_time:,} samples")
        
        try:
            try:
                analysis = extract_dependencies(codebase_dir, func.name, config)
            except Exception as e:
                logger.warning(f"Could not extract dependencies for function {func.name}, skipping...")
                logger.debug(traceback.format_exc())
                continue
            
            if not analysis.functions:
                logger.warning(f"Could not find function {func.name} in codebase. Skipping...")
                continue
            
            logger.info("Generating optimization...")
            try:
                result = optimizer.optimize_function(codebase_dir, func.name, analysis, optimized_count=optimized_count)
            except Exception as e:
                logger.error(f"Error generating optimization for function {func.name}, skipping...")
                logger.debug(traceback.format_exc())
                continue
            
            logger.info(f"Generated function: {result.optimized_function}")
            if result.original_function in result.optimized_function:
                logger.info("Optimization did not change the function. Skipping...")
                continue
            
            logger.info("Applying optimization...")
            try:
                optimizer.apply_optimization(result, codebase_dir)
            except Exception as e:
                logger.error(f"Error applying optimization for function {func.name}, skipping...")
                logger.debug(traceback.format_exc())
                continue
            
            logger.info(f"\nSuccess! Modified file: {result.file_path}, Created branch: {result.branch_name}")
            
            logger.info("Verifying performance improvement...")
            cur_perf = perf_verifier.get_performance()
            new_perf = perf_verifier.get_performance(result.branch_name)
            logger.info(f"Current performance: {cur_perf}")
            logger.info(f"New performance: {new_perf}")

            if not perf_verifier.validate_performance(new_perf) or not perf_verifier.validate_performance(cur_perf):
                # Performance measurement failed
                logger.error("Performance measurement failed, likely due to compilation errors. Skipping...")
                # todo this is where we can re-prompt the LLM to generate a syntactically correct function
            if not perf_verifier.tests_pass(result.branch_name):
                logger.error("Tests failed on new branch. Skipping...")
                continue
        
            logger.info(f"Tests passed on branch {result.branch_name}")
            if perf_verifier.compare_performance(cur_perf, new_perf):
                logger.info("Performance improved!")
                optimized_count += 1
                # make this the new current branch
                os.system(f'git -C {codebase_dir} checkout -q {result.branch_name}')
            
        except Exception as e:
            logger.error(f"Unexpected error optimizing function: {str(e)}")
            logger.debug(traceback.format_exc())
            continue

    final_perf = perf_verifier.get_performance()
    logger.info(f"\nOptimization loop complete.\n\nInitial performance: {original_perf}\nFinal performance: {final_perf}")
    perf_verifier.summarize_improvements(original_perf, final_perf, event="cycles")
    perf_verifier.summarize_improvements(original_perf, final_perf, event="task-clock")
    perf_verifier.summarize_improvements(original_perf, final_perf, event="instructions")



def main():
    parser = argparse.ArgumentParser(
        description='Optimize performance hotspots in a C++ codebase using LLM'
    )
    parser.add_argument('codebase_dir', help='Directory containing the C++ codebase')
    parser.add_argument('stacks_dir', help='Directory containing the folded stack files')
    parser.add_argument('--num-functions', type=int, default=3, help='Number of top functions to optimize (default: 3)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use for optimization (default: gpt-3.5-turbo)')
    parser.add_argument('--provider', default='openai', help='API provider to use for optimization (default: openai)')
    args = parser.parse_args()
    
    setup_logger(args.debug)
    logger = logging.getLogger()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key and not openai_key:
        logger.error("Error: Anthropic API key required")
        logger.error("Set ANTHROPIC_API_KEY environment variable")
        return 1
    
    codebase_dir = os.path.abspath(args.codebase_dir)
    pv = APEHW1(codebase_dir)
    
    try:
        optimize_hotspots(
            codebase_dir,
            args.stacks_dir,
            openai_key if args.provider == 'openai' else anthropic_key,
            pv,
            args.num_functions,
            model=args.model,
            provider=args.provider
        )
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
