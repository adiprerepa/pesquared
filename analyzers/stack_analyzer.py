import os
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass

@dataclass
class FunctionStats:
    name: str
    exclusive_time: int
    call_chain: List[str] = None

    def __repr__(self) -> str:
        if self.call_chain:
            return f"FunctionStats(name='{self.name}', exclusive_time={self.exclusive_time}, call_chain={self.call_chain})"
        return f"FunctionStats(name='{self.name}', exclusive_time={self.exclusive_time})"

class StackAnalyzer:
    KERNEL_PATTERNS = [
        r'^entry_SYSCALL_64', r'^sys_', r'^perf_event_', r'^mutex_',
        r'^raw_spin_', r'^rcu_', r'^futex_', r'^_raw_', r'\[unknown\]$',
    ]

    PROCESS_PREFIXES = [
        'main.exe', 'ffmpeg', 'magick', 'av:', 'dec0:', 'mux0:', 'dmx0:',
        'perf-exec', 'sh',
    ]

    def __init__(self, input_path: str):
        self.input_path = input_path
        self._exclusive_times: Dict[str, int] = {}
        self._is_processed = False
        self._function_data = defaultdict(lambda: {"exclusive_time": 0, "call_chains": {}})
        self.call_graph = nx.DiGraph()

    def _is_kernel_or_unknown_function(self, func_name: str) -> bool:
        return any(re.search(pattern, func_name) for pattern in self.KERNEL_PATTERNS)

    def _parse_folded_stack(self, line: str) -> Tuple[List[str], int]:
        try:
            stack_str, count_str = line.strip().rsplit(None, 1)
            count = int(count_str)
            frames = [frame.strip() for frame in stack_str.split(';')]
            frames = [f for f in frames if f != '[unknown]' and f.strip()]
            return frames, count
        except (ValueError, IndexError):
            return [], 0

    def _process_file(self, filepath: str):
        with open(filepath, 'r') as f:
            for line in f:
                frames, count = self._parse_folded_stack(line)
                if not frames:
                    continue

                leaf_function = frames[-1]
                if not self._is_kernel_or_unknown_function(leaf_function):
                    clean_frames = []
                    for frame in frames:
                        if not clean_frames or clean_frames[-1] != frame:
                            clean_frames.append(frame)

                    call_chain_tuple = tuple(clean_frames)
                    self._function_data[leaf_function]["exclusive_time"] += count

                    chains = self._function_data[leaf_function]["call_chains"]
                    if call_chain_tuple in chains:
                        chains[call_chain_tuple] += count
                    else:
                        chains[call_chain_tuple] = count

                    for i, func in enumerate(clean_frames):
                        if not self.call_graph.has_node(func):
                            self.call_graph.add_node(func, inclusive_time=0, exclusive_time=0)
                        self.call_graph.nodes[func]['inclusive_time'] += count
                    self.call_graph.nodes[leaf_function]['exclusive_time'] += count
                    for parent, child in zip(clean_frames, clean_frames[1:]):
                        if self.call_graph.has_edge(parent, child):
                            self.call_graph[parent][child]['weight'] += count
                        else:
                            self.call_graph.add_edge(parent, child, weight=count)

    def process(self):
        if self._is_processed:
            return

        if os.path.isfile(self.input_path):
            self._process_file(self.input_path)
        else:
            for filename in os.listdir(self.input_path):
                if filename.endswith('.folded'):
                    self._process_file(os.path.join(self.input_path, filename))

        self._exclusive_times = {func: data["exclusive_time"] for func, data in self._function_data.items()}
        self._is_processed = True

    def get_top_functions(self, n: int = 10) -> List[FunctionStats]:
        if not self._is_processed:
            self.process()

        sorted_funcs = sorted(self._exclusive_times.items(), key=lambda x: x[1], reverse=True)
        result = []
        for name, time in sorted_funcs[:n]:
            call_chains = self._function_data[name]["call_chains"]
            if call_chains:
                most_frequent_chain = max(call_chains.items(), key=lambda x: x[1])[0]
                result.append(FunctionStats(name=name, exclusive_time=time, call_chain=list(most_frequent_chain)))
            else:
                result.append(FunctionStats(name=name, exclusive_time=time))
        return result

    def plot(self, figsize=(12, 6)):
        if not self._is_processed:
            self.process()

        path_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        for data in self._function_data.values():
            for chain, cnt in data["call_chains"].items():
                path_counts[chain] += cnt

        if not path_counts:
            print("No data to plot.")
            return

        x_offset: Dict[int, int] = defaultdict(int)
        max_depth = max(len(chain) for chain in path_counts)

        fig, ax = plt.subplots(figsize=figsize)
        for chain, cnt in path_counts.items():
            for depth, func in enumerate(chain):
                x0 = x_offset[depth]
                rect = patches.Rectangle((x0, -depth), cnt, 1)
                ax.add_patch(rect)
                ax.text(x0 + cnt/2, -depth + 0.5, func, ha="center", va="center", fontsize=6, clip_on=True)
                x_offset[depth] += cnt

        ax.set_xlim(0, max(x_offset.values()))
        ax.set_ylim(-max_depth, 1)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze folded stacks for exclusive function times')
    parser.add_argument('input_path', help='Path to a .folded file or directory of .folded files')
    parser.add_argument('-n', '--top_n', type=int, default=10, help='Number of top functions to display')
    parser.add_argument('--show-call-chains', action='store_true', help='Display the call chains for each function')
    parser.add_argument('--plot', action='store_true', help='Plot the flamegraph using matplotlib')
    args = parser.parse_args()

    analyzer = StackAnalyzer(args.input_path)
    top_functions = analyzer.get_top_functions(args.top_n)

    print(f"\nTop {args.top_n} functions by exclusive time:")
    print("-" * 80)
    for idx, func in enumerate(top_functions, 1):
        print(f"{idx}. {func.name:<40} {func.exclusive_time:>15,} samples")
        if args.show_call_chains and func.call_chain:
            print(f"   Call chain: {' -> '.join(func.call_chain)}")
    print("-" * 80)

    if args.plot:
        analyzer.plot()

if __name__ == "__main__":
    main()
