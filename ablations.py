import argparse
from collections import defaultdict
from typing import List
from analyzers.repo_analyzer import RepoAnalyzer, in_repo_and_not_variable
from optimizers.repo_optimizer import RepoOptimizer
import os
from analyzers.stack_analyzer import StackAnalyzer
import pandas as pd
import logging
import asyncio
from verifiers import hw1, hw2
import networkx as nx
from tqdm.asyncio import tqdm

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

    # pv = hw1.APEHW1(args.codebase_dir)

    pv = hw2.APEHW2(args.codebase_dir)
    logging.getLogger().info("Logger initialized")
    analyzer = RepoAnalyzer.from_path(args.codebase_dir, pv=pv)
    subgraph = analyzer.induce_subgraph(node_filter=in_repo_and_not_variable)
    levels = {}

    # --- Level Computation ---
    if nx.is_directed_acyclic_graph(subgraph):
        # Regular topological sort
        for node in nx.topological_sort(subgraph):
            preds = list(subgraph.predecessors(node))
            levels[node] = 0 if not preds else 1 + max(levels[p] for p in preds)
    else:
        print("⚠️ Graph is not a DAG. Using SCC condensation for level estimation.")
        cycle = list(nx.simple_cycles(subgraph))
        if cycle:
            print(f"⚠️ Found cycles: {cycle}")
        sccs = list(nx.strongly_connected_components(subgraph))
        scc_map = {node: idx for idx, comp in enumerate(sccs) for node in comp}
        condensed = nx.condensation(subgraph, sccs)

        scc_levels = {}
        for scc_node in nx.topological_sort(condensed):
            preds = list(condensed.predecessors(scc_node))
            scc_levels[scc_node] = 0 if not preds else 1 + max(scc_levels[p] for p in preds)

        # Map back to node-level
        for node in subgraph.nodes:
            scc_index = scc_map[node]
            levels[node] = scc_levels[scc_index]
    stack_analyzer = StackAnalyzer(analyzer.perfstacks_dir)
    top_functions = stack_analyzer.get_top_functions(15)
    # make top functions reverse order 
    # top_functions = top_functions[::-1]

    measure_perf_count = 5
    
    
    output_csv = 'optimization_results.csv'
    write_header = True

    existing_functions = set()
    if os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            if 'function' in existing_df.columns:
                existing_functions = set(existing_df['function'].dropna().unique())
        except Exception as e:
            print(f"Warning: Could not load {output_csv}: {e}")
    
    baseline_perf, tests_pass = pv.get_performance(n_iters=measure_perf_count, fn_list=[f.name for f in top_functions])
    # for k in baseline_perf:
    #     for k2 in baseline_perf[k]:
    #         baseline_perf[k][k2] = float(baseline_perf[k][k2])
    print(f"[ablations] baseline_perf (tests_pass {tests_pass}): {baseline_perf}")
    if isinstance(pv, hw1.APEHW1):
        assert baseline_perf['globe']['task-clock'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['pianoroom']['task-clock'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['globe']['cycles'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['pianoroom']['cycles'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['globe']['instructions'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['pianoroom']['instructions'] > 0, "Baseline performance is zero or negative"
    elif isinstance(pv, hw2.APEHW2):
        assert baseline_perf['diabetes']['task-clock'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['diabetes']['cycles'] > 0, "Baseline performance is zero or negative"
        assert baseline_perf['diabetes']['instructions'] > 0, "Baseline performance is zero or negative"
    results = []
    # (model, provider)
    openai = [('gpt-4o-mini-2024-07-18', 'openai'), ('gpt-4o-2024-08-06', 'openai')]
    claude = [('claude-3-5-sonnet-20241022', 'anthropic'), ('claude-3-7-sonnet-20250219', 'anthropic')]
    gemini = [('gemini-1.5-flash-8b', 'google'), ('gemini-1.5-flash', 'google'), ('gemini-1.5-pro', 'google')]
    # gemini = [('gemini-1.5-flash', 'google')]
    # llms = [openai[0]] + [claude[0]] + [gemini[2]]
    llms = gemini
    # llms = [('gemini-1.5-flash-8b', 'google')]
    call_depth = [0, 1] #[0, 1]
    obfuscation_tiers = [0, 3] #[0, 3]
    

    print(f"Existing functions in {output_csv}: {existing_functions}")
    for f in top_functions:
        call_chain = f.call_chain[::-1]
        for name in call_chain:
            print(f"Optimizing function {name}")
            if name not in analyzer.symbol_to_signatures:
                print(f"Skipping function {name} as it is not in symbol_to_signatures.")
                continue
            function = analyzer.symbol_to_signatures[name][0]
            if function in existing_functions:
                print(f"[skip] Skipping function {function} — already optimized in {output_csv}")
                continue 
            params = []
            for model, provider in llms:
                for callee_depth in call_depth:
                    for obfuscation_tier in obfuscation_tiers:
                        if 'missed_optimizations' in analyzer.nodes[function]:
                            include_clang_remarks_options = [False, True]
                        else:
                            include_clang_remarks_options = [False]
                        for include_clang_remarks in include_clang_remarks_options:
                            params.append((include_clang_remarks, obfuscation_tier, callee_depth, model, provider))
            # run repo_opt.build_optimization_prompt_callable(...) for all params in parallel using asyncio
            print(f"Running {len(params)} configurations for function {function}.\nparams: {params}")
            tasks = []
            repo_opts: List[RepoOptimizer] = []
            for include_clang_remarks, obfuscation_tier, callee_depth, model, provider in params:
                repo_opt = RepoOptimizer(model, provider, analyzer, pv)
                repo_opts.append(repo_opt)
                tasks.append(repo_opt.build_optimization_prompt_callable(function,
                    obfuscation_tier=obfuscation_tier, callee_depth=callee_depth,
                    node_filter=in_repo_and_not_variable, include_clang_remarks=include_clang_remarks,
                    model=model, provider=provider))
            # run all tasks in parallel
            results = await tqdm.gather(*tasks, desc="Optimizing functions", total=len(tasks))
            implementations = [repo_opts[i].implement_optimization(*result) for i, result in enumerate(results)]
            branches = [result['branch_name'] for result in implementations if result and result['model_optimized']]
            branch_to_model_provider = {impl['branch_name']: (impl['model'], impl['provider']) for impl in implementations if impl and impl['model_optimized']}
            branch_to_clang_remarks = {impl['branch_name']: impl['include_clang_remarks'] for impl in implementations if impl and impl['model_optimized']}
            branch_to_obfuscation_tier = {impl['branch_name']: impl['obfuscation_tier'] for impl in implementations if impl and impl['model_optimized']}
            branch_to_callee_depth = {impl['branch_name']: impl['callee_depth'] for impl in implementations if impl and impl['model_optimized']}

            if len(branches) == 0:
                print(f"Skipping function {function} as no branches were created.")
                continue
            print("================")
            print(f"branches: {branches}")
            print("================")
            all_perfs = pv.get_all_performance_parallel(branches=branches, n_iters=measure_perf_count, fn=name)
            print(all_perfs)

            for branch, perf in all_perfs.items():
                print(f"perf for branch {branch}: {perf}")
                if perf is None or not pv.validate_performance(perf):
                    print(f"Skipping branch {branch} as performance is invalid.")
                    continue
                model, provider = branch_to_model_provider[branch]
                def get_improvement(baseline: float, current: float) -> float:
                    """Return percent improvement from baseline to current."""
                    if baseline == 0:
                        return 0.0  # avoid division by zero
                    return 100.0 * (baseline - current) / baseline

                def get_float(d: dict, *keys: str) -> float:
                    """Safely fetch nested float value."""
                    val = d
                    for key in keys:
                        val = val[key]
                    return float(val)
                
                def get_baseline_float(d: dict, fn: str, region: str) -> float:
                    target_fns = baseline_perf[region].get('target_fns', {})
                    if name in target_fns and target_fns[name] > 0:
                        baseline = target_fns[name]
                        return baseline
                    else:
                        return -1

                def maybe_get_target_fn_improvement(baseline_perf, perf, region: str, name: str) -> dict:
                    """Compute target function improvement if available."""
                    results = {}
                    target_fns = baseline_perf[region].get('target_fns', {})
                    if name in target_fns and target_fns[name] > 0:
                        baseline = target_fns[name]
                        current = get_float(perf, region, 'target_fn_cycles')
                        results[f'{region}_cycles_target_function_improvement'] = get_improvement(baseline, current)
                    else:
                        results[f'{region}_cycles_target_function_improvement'] = 0
                    return results

                # Build the row dictionary
                row_data = {
                    'function': function,
                    'function_relative_level': levels[function],
                    'function_length': len(analyzer.nodes[function]['code'].splitlines()),
                    'include_clang_remarks': branch_to_clang_remarks[branch],
                    'obfuscation_tier': branch_to_obfuscation_tier[branch],
                    'callee_depth': branch_to_callee_depth[branch],
                    'model': model,
                    'provider': provider,
                    
                    'globe_baseline_cycles': get_float(baseline_perf, 'globe', 'cycles'),
                    'globe_baseline_runtime': get_float(baseline_perf, 'globe', 'task-clock'),
                    'globe_cycles': get_float(perf, 'globe', 'cycles'),
                    'globe_runtime': get_float(perf, 'globe', 'task-clock'),
                    'globe_instructions': get_float(perf, 'globe', 'instructions'),
                    'globe_cycles_target_function': get_float(perf, 'globe', 'target_fn_cycles'),
                    'globe_cycles_target_function_baseline': get_baseline_float(baseline_perf, function, 'globe'),
                    

                    'pianoroom_cycles': get_float(perf, 'pianoroom', 'cycles'),
                    'pianoroom_runtime': get_float(perf, 'pianoroom', 'task-clock'),
                    'pianoroom_instructions': get_float(perf, 'pianoroom', 'instructions'),
                    'pianoroom_baseline_cycles': get_float(baseline_perf, 'pianoroom', 'cycles'),
                    'pianoroom_baseline_runtime': get_float(baseline_perf, 'pianoroom', 'task-clock'),
                    'pianoroom_cycles_target_function': get_float(perf, 'pianoroom', 'target_fn_cycles'),   


                    'globe_cycles_improvement': get_improvement(get_float(baseline_perf, 'globe', 'cycles'), get_float(perf, 'globe', 'cycles')),
                    'globe_runtime_improvement': get_improvement(get_float(baseline_perf, 'globe', 'task-clock'), get_float(perf, 'globe', 'task-clock')),
                    'globe_instructions_improvement': get_improvement(get_float(baseline_perf, 'globe', 'instructions'), get_float(perf, 'globe', 'instructions')),

                    'pianoroom_cycles_improvement': get_improvement(get_float(baseline_perf, 'pianoroom', 'cycles'), get_float(perf, 'pianoroom', 'cycles')),
                    'pianoroom_runtime_improvement': get_improvement(get_float(baseline_perf, 'pianoroom', 'task-clock'), get_float(perf, 'pianoroom', 'task-clock')),
                    'pianoroom_instructions_improvement': get_improvement(get_float(baseline_perf, 'pianoroom', 'instructions'), get_float(perf, 'pianoroom', 'instructions')),

                    'branch_name': branch,
                    'tests_passed': bool(perf['tests_passed']),
                }

                # Add target function improvements if available
                row_data.update(maybe_get_target_fn_improvement(baseline_perf, perf, 'pianoroom', name))
                row_data.update(maybe_get_target_fn_improvement(baseline_perf, perf, 'globe', name))

                # Finally create the DataFrame
                row = pd.DataFrame([row_data])
                    
                # Append row to CSV immediately
                row.to_csv(output_csv, mode='a', header=write_header, index=False)
                write_header = False  # Only write header once
    
if __name__ == "__main__":
    asyncio.run(main())