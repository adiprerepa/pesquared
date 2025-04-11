#!/usr/bin/env python3

import argparse
import os
import traceback
import logging
import signal
import sys
from dotenv import load_dotenv

from analyzers.clang_remark_analyzer import index_remarks_by_function
from analyzers.stack_analyzer import StackAnalyzer
from analyzers.dependency_extractor import extract_dependencies, DependencyExtractorConfig, format_analysis_output, \
    demangle_name
from optimizers.function_optimizer import FunctionOptimizer, OptimizationResult
from verifiers.base_verifier import PerformanceVerifier
from verifiers.hw1 import APEHW1
from verifiers.hw2 import APEHW2
from utils import git_utils

# Load environment variables from .env file
load_dotenv()

def setup_logger(debug: bool, codebase_dir: str = '.'):
    """Set up logger with specified debug level and git branch prefix."""
    level = logging.DEBUG if debug else logging.INFO
    git_utils.setup_branch_logging(codebase_dir, level)

def optimize_hotspots(codebase_dir: str, stacks_dir: str, api_key: str, perf_verifier: PerformanceVerifier, num_functions: int = 3, model="gpt-4o", provider="openai", correction_attempts: int = 2, temperature=0.7, max_depth: int = None) -> None:
    """
    Optimize the top hotspot functions and their call chains in a loop.

    Args:
        codebase_dir: Directory containing the C++ codebase
        stacks_dir: Directory containing the folded stack files
        api_key: API key for the LLM provider
        perf_verifier: Performance verification instance
        num_functions: Number of top call chains to optimize (default: 3)
        model: LLM model to use
        provider: LLM provider (openai or anthropic)
        correction_attempts: Maximum number of attempts to correct compilation errors (default: 2)
        max_depth: Maximum depth in the call chain to explore (default: None, which means no limit)
    """

    logger = logging.getLogger()
    stack_analyzer = StackAnalyzer(stacks_dir)
    optimizer = FunctionOptimizer(api_key=api_key, model=model, provider=provider)
    config = DependencyExtractorConfig(
        include_function_locations=True,
        include_type_locations=True
    )

    functions_iter = stack_analyzer.iter_functions()
    optimized_chains = 0

    logger.info(f"🚀 Starting optimization loop for top {num_functions} hotspot call chains...")

    original_perf, original_pass = perf_verifier.get_performance()
    optimized_functions = set()  # Track all optimized functions to avoid repeats

    while optimized_chains < num_functions:
        try:
            func = next(functions_iter)
        except StopIteration:
            logger.info("🔍 No more functions to analyze")
            break

        if "[unknown]" in func.name or not func.name.strip():
            continue

        # Skip if we've already processed this function
        if func.name in optimized_functions:
            continue

        # Log information about the current hotspot
        logger.info(f"\n🔥 Analyzing hotspot call chain #{optimized_chains + 1}:")
        logger.info(f"📊 Leaf function: {func.name}")
        logger.info(f"⏱️ Exclusive time: {func.exclusive_time:,} samples")

        # Process the call chain if available
        call_chain = func.call_chain if func.call_chain else [func.name]

        # If max_depth is set, limit the call chain to the specified depth
        if max_depth is not None and len(call_chain) > max_depth:
            logger.info(f"⚠️ Call chain depth ({len(call_chain)}) exceeds max depth ({max_depth}), truncating...")
            # Keep the leaf function and max_depth-1 ancestors (to make a total of max_depth functions)
            call_chain = call_chain[-(max_depth):]

        if len(call_chain) > 1:
            logger.info(f"🔗 Call chain: {' -> '.join(call_chain)}")

        # Optimize each function in the call chain, starting from the root of the call chain
        # This ensures that we optimize parent functions before their children
        for idx, function_name in enumerate(reversed(call_chain)):
            # Skip if we've already optimized this function
            if function_name in optimized_functions:
                logger.info(f"✅ Function {function_name} already optimized, skipping...")
                continue

            if "[unknown]" in function_name or not function_name.strip():
                logger.info(f"❓ Skipping unknown function in call chain: {function_name}")
                continue

            logger.info(f"\n⚙️ Optimizing function {idx+1}/{len(call_chain)} in call chain: {function_name}")

            try:
                try:
                    analysis = extract_dependencies(codebase_dir, function_name, config)
                except Exception as e:
                    logger.warning(f"⚠️ Could not extract dependencies for function {function_name}, skipping...")
                    logger.debug(traceback.format_exc())
                    continue

                if not analysis.functions:
                    logger.warning(f"🔎 Could not find function {function_name} in codebase. Skipping...")
                    continue

                remarks = perf_verifier.get_remarks()
                demangled_fn = demangle_name(function_name)
                index = index_remarks_by_function(remarks)
                logger.info(f"index keys: {index.keys()}")
                if demangled_fn in index:
                    function_remarks = index[demangled_fn]
                    logger.info(f"found {len(function_remarks)} remarks for {demangled_fn}")
                else:
                    function_remarks = []
                    logger.info(f"no remarks for {demangled_fn}")

                logger.info("🧠 Generating optimization...")
                try:
                    # Ensure we use mangled names if available
                    # All our core functions now handle mangled names properly
                    result = optimizer.optimize_function(
                        codebase_dir,
                        function_name,  # can be mangled or unmangled
                        analysis,
                        optimized_count=optimized_chains,
                        temperature=temperature,
                        call_chain=call_chain,
                        position_in_chain=len(call_chain) - 1 - idx,  # Since we're iterating in reverse,
                        function_remarks=function_remarks
                    )
                except Exception as e:
                    logger.error(f"❌ Error generating optimization for function {function_name}, skipping...")
                    logger.debug(traceback.format_exc())
                    continue

                logger.info(f"✨ Generated function: {result.optimized_function}")
                if result.original_function in result.optimized_function:
                    logger.info("🔄 Optimization did not change the function. Skipping...")
                    continue

                logger.info("🔧 Applying optimization...")
                try:
                    optimizer.apply_optimization(result, codebase_dir)
                except Exception as e:
                    logger.error(f"❌ Error applying optimization for function {function_name}, skipping...")
                    logger.debug(traceback.format_exc())
                    continue

                logger.info(f"\n✅ Success! Modified file: {result.file_path}, Created branch: {result.branch_name}")


                # Use the configured max number of attempts to fix compilation issues
                max_correction_attempts = correction_attempts
                current_attempt = 0
                current_result = result

                # Initialize performance variables before the loop to avoid UnboundLocalError
                logger.info("📊 Verifying performance improvement...")
                cur_perf, cur_pass = perf_verifier.get_performance()
                new_perf, new_pass = perf_verifier.get_performance(result.branch_name)
                logger.info(f"🧪 Test validation: current branch={cur_pass}, new branch={new_pass}\nPerformance:")
                perf_verifier.summarize_improvements(cur_perf, new_perf, event="cycles")

                # Always check performance once, regardless of max_correction_attempts
                if perf_verifier.validate_performance(new_perf) and perf_verifier.validate_performance(cur_perf):
                    # Performance validation successful, check if tests pass
                    if not new_pass:
                        logger.error("❌ Tests failed on branch. Skipping...")
                    else:
                        logger.info(f"✅ Tests passed on branch {current_result.branch_name}")
                        if perf_verifier.compare_performance(cur_perf, new_perf):
                            logger.info(f"Performance improved ({function_name})! Making {current_result.branch_name} the new current branch🚀")
                            # Add to set of optimized functions
                            optimized_functions.add(function_name)


                            # Make this the new current branch
                            git_utils.checkout_branch(current_result.branch_name, codebase_dir)
                else:
                    logger.error("❌ Performance validation failed. Skipping...")
                    logger.error(f"  Current branch: {git_utils.get_current_branch(codebase_dir)}")

            except Exception as e:
                logger.error(f"💥 Unexpected error optimizing function: {str(e)}")
                logger.debug(traceback.format_exc())
                continue

        # Count this call chain as processed regardless of how many functions were successfully optimized
        optimized_chains += 1

    final_perf, final_pass = perf_verifier.get_performance()
    # logger.info(f"\nOptimization loop complete.\n\nInitial performance: {original_perf}\nFinal performance: {final_perf}")
    logger.info(f"🏁 Loop complete. Test (original_pass: {original_pass}, final_pass: {final_pass}, Total functions optimized: {len(optimized_functions)}")
    perf_verifier.summarize_improvements(original_perf, final_perf, event="cycles")
    perf_verifier.summarize_improvements(original_perf, final_perf, event="task-clock")
    perf_verifier.summarize_improvements(original_perf, final_perf, event="instructions")

    # Display token usage and cost
    from optimizers.function_optimizer import token_tracker
    logger.info(token_tracker)




def optimize_single_function(codebase_dir: str, function_name: str, api_key: str, perf_verifier: PerformanceVerifier, model="gpt-4o", provider="openai", correction_attempts: int = 2, temperature=0.7) -> None:
    """
    Optimize a single specified function without stack analysis.

    Args:
        codebase_dir: Directory containing the C++ codebase
        function_name: Name of the function to optimize
        api_key: API key for the LLM provider
        perf_verifier: Performance verification instance
        model: LLM model to use
        provider: LLM provider (openai or anthropic)
        correction_attempts: Maximum number of attempts to correct compilation errors
        temperature: Temperature for sampling from the model
    """

    logger = logging.getLogger()
    optimizer = FunctionOptimizer(api_key=api_key, model=model, provider=provider)
    config = DependencyExtractorConfig(
        include_function_locations=True,
        include_type_locations=True
    )

    logger.info(f"🚀 Optimizing single function: {function_name}")

    original_perf, original_pass = perf_verifier.get_performance()


    try:
        # Extract dependencies for the function (using mangled name)
        try:
            analysis = extract_dependencies(codebase_dir, function_name, config)
        except Exception as e:
            logger.error(f"⚠️ Could not extract dependencies for function {function_name}")
            logger.debug(traceback.format_exc())
            return

        if not analysis.functions:
            logger.error(f"🔎 Could not find function {function_name} in codebase.")
            return

        logger.info("🧠 Generating optimization...")
        try:
            # Optimize the function - handles both mangled and unmangled names
            result = optimizer.optimize_function(
                codebase_dir,
                function_name,  # Can be mangled or unmangled
                analysis,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"❌ Error generating optimization for function {function_name}")
            logger.debug(traceback.format_exc())
            return

        logger.info(f"✨ Generated function: {result.optimized_function}")
        if result.original_function in result.optimized_function:
            logger.info("🔄 Optimization did not change the function. Skipping...")
            return

        logger.info("🔧 Applying optimization...")
        try:
            optimizer.apply_optimization(result, codebase_dir)
        except Exception as e:
            logger.error(f"❌ Error applying optimization for function {function_name}")
            logger.debug(traceback.format_exc())
            return

        logger.info(f"\n✅ Success! Modified file: {result.file_path}, Created branch: {result.branch_name}")

        # Verify performance improvement
        logger.info("📊 Verifying performance improvement...")
        cur_perf, cur_pass = perf_verifier.get_performance()
        new_perf, new_pass = perf_verifier.get_performance(result.branch_name)
        logger.info(f"🧪 Test validation: current branch={cur_pass}, new branch={new_pass}\nPerformance:")
        perf_verifier.summarize_improvements(cur_perf, new_perf, event="cycles")

        if perf_verifier.validate_performance(new_perf) and perf_verifier.validate_performance(cur_perf):
            if not new_pass:
                logger.error("❌ Tests failed on branch. Skipping...")
            else:
                logger.info(f"✅ Tests passed on branch {result.branch_name}")
                if perf_verifier.compare_performance(cur_perf, new_perf):
                    logger.info(f"Performance improved! Making {result.branch_name} the new current branch🚀")


                    # make this the new current branch
                    git_utils.checkout_branch(result.branch_name, codebase_dir)
        else:
            logger.error("❌ Performance validation failed. Skipping...")

    except Exception as e:
        logger.error(f"💥 Unexpected error optimizing function: {str(e)}")
        logger.debug(traceback.format_exc())

    # Display token usage and cost
    from optimizers.function_optimizer import token_tracker
    logger.info(token_tracker)


def main():
    parser = argparse.ArgumentParser(
        description='Optimize performance hotspots in a C++ codebase using LLM'
    )
    parser.add_argument('codebase_dir', help='Directory containing the C++ codebase')
    parser.add_argument('stacks_dir', help='Directory containing the folded stack files')
    parser.add_argument('--function', help='Optimize a specific function instead of using stack analysis')
    parser.add_argument('--num-functions', type=int, default=3, help='Number of top functions to optimize (default: 3)')
    parser.add_argument('--correction-attempts', type=int, default=2, help='Maximum number of attempts to correct compilation errors (default: 2)')
    parser.add_argument('--max-depth', type=int, help='Maximum depth of call chain to explore from the leaf hotspot (default: no limit)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='LLM model to use for optimization (default: gpt-3.5-turbo)')
    parser.add_argument('--provider', default='openai', help='API provider to use for optimization (default: openai)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling from the model (default: 0.7)')
    parser.add_argument('--obfuscate', type=bool, default=False, help='Obfuscate the code before sending to the model (default: False)')
    args = parser.parse_args()

    if args.codebase_dir.startswith('https://'):
        print("🔗 Cloning codebase from URL...")
        # dir is a git URL
        # Clone the repo to a tmp/user/repo and set codebase_dir to that
        from utils.git_utils import clone_repo
        repo = clone_repo(args.codebase_dir)
        codebase_dir = os.path.abspath(repo.working_tree_dir)
        print(f"Cloned repo to {codebase_dir}")
    else:
        codebase_dir = os.path.abspath(args.codebase_dir)
    if not os.path.exists(args.stacks_dir):
        args.stacks_dir = os.path.join(codebase_dir, args.stacks_dir)
    setup_logger(args.debug, codebase_dir)
    logger = logging.getLogger()

    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_key and not openai_key:
        logger.error("🔑 Error: API key required")
        logger.error("🔑 Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return 1

    # Select the appropriate API key based on provider
    api_key = openai_key if args.provider == 'openai' else anthropic_key

    # pv = APEHW1(codebase_dir)
    pv = APEHW2(codebase_dir)

    # Store original performance for Ctrl+C handler
    original_perf, original_pass = pv.get_performance()

    remarks = pv.get_remarks()
    logger.info(f"Got {len(remarks)} remarks on the current branch.")

    # Setup Ctrl+C handler
    interrupt_count = 0

    def signal_handler(sig, frame):
        nonlocal interrupt_count
        interrupt_count += 1

        if interrupt_count == 1:
            logger.info("\n🛑 Process interrupted by user (Ctrl+C)")
            logger.info("Press Ctrl+C again to exit immediately without performance summary")
            try:
                final_perf, final_pass = pv.get_performance()
                logger.info("📊 Performance Summary on Interrupt:")
                pv.summarize_improvements(original_perf, final_perf, event="cycles")
                pv.summarize_improvements(original_perf, final_perf, event="task-clock")
                pv.summarize_improvements(original_perf, final_perf, event="instructions")

                # Display token usage and cost
                from optimizers.function_optimizer import token_tracker
                logger.info("📝 Token usage and cost:")
                logger.info(token_tracker)


                sys.exit(0)
            except Exception as e:
                logger.error(f"Error generating performance summary: {str(e)}")
                sys.exit(1)
        else:
            logger.info("\n🛑 Exiting immediately due to multiple interrupts")
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Check if a specific function was specified
        if args.function:
            # Skip stack analysis and optimize the single function
            optimize_single_function(
                codebase_dir,
                args.function,
                api_key,
                pv,
                model=args.model,
                provider=args.provider,
                correction_attempts=args.correction_attempts,
                temperature=args.temperature
            )
        else:
            # Use stack analysis to find and optimize hotspots
            optimize_hotspots(
                codebase_dir,
                args.stacks_dir,
                api_key,
                pv,
                args.num_functions,
                model=args.model,
                provider=args.provider,
                correction_attempts=args.correction_attempts,
                temperature=args.temperature,
                max_depth=args.max_depth
            )
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
