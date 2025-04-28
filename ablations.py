import argparse
from analyzers.repo_analyzer import RepoAnalyzer, is_kind, is_function, is_variable, is_type, in_repo_and_not_variable
from optimizers.repo_optimizer import RepoOptimizer
import os
from analyzers.stack_analyzer import StackAnalyzer
import pandas as pd
import logging
import asyncio
from verifiers import hw1

async def main():
    parser = argparse.ArgumentParser(
        description='Optimize performance hotspots in a C++ codebase using LLM'
    )
    parser.add_argument('codebase_dir', help='Directory containing the C++ codebase.')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')

    # parser.add_argument('stacks_dir', help='Directory containing the folded stacks.')
    # parser.add_argument('--num-functions', type=int, default=3, help='Number of top functions to optimize (default: 3)')
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger().info("Logger initialized")
    analyzer = RepoAnalyzer.from_path(args.codebase_dir)
    stack_analyzer = StackAnalyzer(analyzer.perfstacks_dir)
    top_functions = stack_analyzer.get_top_functions(3)
    pv = hw1.APEHW1(args.codebase_dir)
    
    # (model, provider)
    output_csv = 'optimization_results.csv'
    write_header = not os.path.exists(output_csv)
    
    baseline_perf = pv.get_performance()
    print(f"[ablations] baseline_perf: {baseline_perf}")
    # return
    results = []
    # ('claude-3-7-sonnet-20250219', 'anthropic'), ('claude-3-5-sonnet-20241022', 'anthropic')
    # llms = [('claude-3-5-sonnet-20241022', 'anthropic'), ('claude-3-7-sonnet-20250219', 'anthropic')]
    llms = [('gemini-2.0-flash', 'google')]
    for f in top_functions:
        call_chain = f.call_chain
        for name in call_chain:
            if name not in analyzer.symbol_to_signatures: continue
            function = analyzer.symbol_to_signatures[name][0]
            for include_clang_remarks in [True, False]:
                for obfuscation_tier in [0, 3]:
                    for callee_depth in [0, 1]:
                        for model, provider in llms:
                            repo_opt = RepoOptimizer(model, provider, analyzer, pv)
                            result = await repo_opt.optimize_function(
                                function, obfuscation_tier=obfuscation_tier, callee_depth=callee_depth,
                                node_filter=in_repo_and_not_variable, include_clang_remarks=include_clang_remarks, samples=1
                            )
                            print(f"[ablations] result: {result}")
                            if not result or not result.get('model_optimized', False) or len(result['globe']) == 0:
                                print(f"skipping sweep for {function} with model {model} and provider {provider} and obfuscation_tier {obfuscation_tier} and callee_depth {callee_depth}")
                                continue
                            
                            row = pd.DataFrame([{
                                'function': function,
                                'include_clang_remarks': include_clang_remarks,
                                'obfuscation_tier': obfuscation_tier,
                                'callee_depth': callee_depth,
                                'model': model,
                                'provider': provider,
                                'globe_cycles': float(result['globe']['cycles']),
                                'globe_runtime': float(result['globe']['task-clock']),
                                'globe_instructions': float(result['globe']['instructions']),
                                'pianoroom_cycles': float(result['pianoroom']['cycles']),
                                'pianoroom_runtime': float(result['pianoroom']['task-clock']),
                                'pianoroom_instructions': float(result['pianoroom']['instructions']),
                                'elephant_cycles': float(result['elephant']['cycles']),
                                'elephant_runtime': float(result['elephant']['task-clock']),
                                'elephant_instructions': float(result['elephant']['instructions']),
                                'tests_pass': result['tests_pass'],
                                'branch_name': result['branch_name'],
                                # 'insights': result['insights'],
                            }])

                            # # Append row to CSV immediately
                            row.to_csv(output_csv, mode='a', header=write_header, index=False)
                            write_header = False  # Only write header once
    
if __name__ == "__main__":
    asyncio.run(main())