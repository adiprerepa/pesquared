#!/usr/bin/env python3

import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Iterator
from dataclasses import dataclass
import argparse
import re

@dataclass
class FunctionStats:
    """Statistics for a single function or call chain."""
    name: str
    exclusive_time: int
    call_chain: List[str] = None

    def __repr__(self) -> str:
        if self.call_chain:
            return f"FunctionStats(name='{self.name}', exclusive_time={self.exclusive_time}, call_chain={self.call_chain})"
        return f"FunctionStats(name='{self.name}', exclusive_time={self.exclusive_time})"

class StackAnalyzer:
    """Analyzer for folded stack traces."""
    
    # Patterns for functions to exclude
    KERNEL_PATTERNS = [
        r'^entry_SYSCALL_64',
        r'^sys_',
        r'^perf_event_',
        r'^mutex_',
        r'^raw_spin_',
        r'^rcu_',
        r'^futex_',
        r'^_raw_',
        r'^\[unknown\]$',  # Only exclude if entire name is [unknown]
    ]
    
    # Known process prefixes to strip
    PROCESS_PREFIXES = [
        'main.exe',
        'ffmpeg',
        'magick',
        'av:',
        'dec0:',
        'mux0:',
        'dmx0:',
        'perf-exec',
        'sh',
    ]
    
    def __init__(self, input_path: str, root_node: str = None):
        """
        Initialize the analyzer with either a directory of folded files or a single file.
        
        Args:
            input_path: Path to either a directory containing .folded files or a single .folded file
            root_node: If set, only accept call chains that originate with this root node
        """
        self.input_path = input_path
        self.root_node = root_node
        self._exclusive_times: Dict[str, int] = {}
        self._is_processed = False

    def _is_kernel_or_unknown_function(self, func_name: str) -> bool:
        """Check if the function is a kernel function or unknown."""
        return any(re.search(pattern, func_name) for pattern in self.KERNEL_PATTERNS)

    def _clean_function_name(self, func_name: str) -> str:
        """Clean the function name by removing process prefixes while preserving mangling."""
        for prefix in self.PROCESS_PREFIXES:
            if func_name.startswith(prefix + ';'):
                func_name = func_name[len(prefix) + 1:]
                break
        parts = func_name.split(';')
        if parts:
            func_name = parts[-1].strip()
        return func_name

    def _parse_folded_stack(self, line: str) -> Tuple[List[str], int]:
        """
        Parse a single line of folded stack into stack frames and sample count.
        """
        try:
            stack_str, count_str = line.strip().rsplit(None, 1)
            count = int(count_str)
            frames = [frame.strip() for frame in stack_str.split(';')]
            return frames, count
        except (ValueError, IndexError):
            return [], 0

    def _process_file(self, filepath: str) -> Dict[str, Dict[str, Union[int, List[str]]]]:
        """Process a single folded stack file."""
        function_data = defaultdict(lambda: {"exclusive_time": 0, "call_chains": {}})
        
        with open(filepath, 'r') as f:
            for line in f:
                frames, count = self._parse_folded_stack(line)
                if not frames:
                    continue

                # Root node filtering
                if self.root_node and (frames[0] != self.root_node):
                    continue
                
                # Remove root node prefix (if matches a known process)
                if frames[0] in self.PROCESS_PREFIXES:
                    frames = frames[1:]

                # Remove [unknown] frames
                frames = [f for f in frames if f != '[unknown]' and f.strip()]
                # print(f"Parsed line: {line.strip()} -> Frames: {frames}, Count: {count}")

                if not frames:
                    continue
                
                leaf_function = frames[-1]
                # print(f"Processing function: {leaf_function} with count: {count}")
                
                if not self._is_kernel_or_unknown_function(leaf_function):
                    clean_frames = []
                    for frame in frames:
                        if not clean_frames or clean_frames[-1] != frame:
                            clean_frames.append(frame)
                    
                    call_chain_tuple = tuple(clean_frames)
                    
                    function_data[leaf_function]["exclusive_time"] += count
                    
                    if call_chain_tuple in function_data[leaf_function]["call_chains"]:
                        function_data[leaf_function]["call_chains"][call_chain_tuple] += count
                    else:
                        function_data[leaf_function]["call_chains"][call_chain_tuple] = count
        
        return function_data

    def process(self) -> None:
        """Process all stack files and compute exclusive times and call chains."""
        if self._is_processed:
            return
        # print(f"Processing input path: {self.input_path}")

        self._function_data = defaultdict(lambda: {"exclusive_time": 0, "call_chains": {}})
        
        if os.path.isfile(self.input_path):
            file_data = self._process_file(self.input_path)
            for func, data in file_data.items():
                self._function_data[func]["exclusive_time"] += data["exclusive_time"]
                for call_chain, count in data["call_chains"].items():
                    if call_chain in self._function_data[func]["call_chains"]:
                        self._function_data[func]["call_chains"][call_chain] += count
                    else:
                        self._function_data[func]["call_chains"][call_chain] = count
        else:
            for dirpath, _, filenames in os.walk(self.input_path):
                for filename in filenames:
                    if not filename.endswith('.folded'):
                        continue

                    # print(f"Processing file: {filename}")
                    filepath = os.path.join(dirpath, filename)
                    file_data = self._process_file(filepath)

                    for func, data in file_data.items():
                        self._function_data[func]["exclusive_time"] += data["exclusive_time"]
                        for call_chain, count in data["call_chains"].items():
                            if call_chain in self._function_data[func]["call_chains"]:
                                self._function_data[func]["call_chains"][call_chain] += count
                            else:
                                self._function_data[func]["call_chains"][call_chain] = count
        
        self._exclusive_times = {func: data["exclusive_time"] for func, data in self._function_data.items()}
        self._is_processed = True

    def get_top_functions(self, n: int = 10) -> List[FunctionStats]:
        """Get the top N functions by exclusive time."""
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

    def iter_functions(self) -> Iterator[FunctionStats]:
        """Iterate over all functions in order of exclusive time."""
        if not self._is_processed:
            self.process()
            
        sorted_funcs = sorted(self._exclusive_times.items(), key=lambda x: x[1], reverse=True)
        
        for name, time in sorted_funcs:
            call_chains = self._function_data[name]["call_chains"]
            if call_chains:
                most_frequent_chain = max(call_chains.items(), key=lambda x: x[1])[0]
                yield FunctionStats(name=name, exclusive_time=time, call_chain=list(most_frequent_chain))
            else:
                yield FunctionStats(name=name, exclusive_time=time)

    def get_function_stats(self, function_name: str) -> Union[FunctionStats, None]:
        """Get stats for a specific function."""
        if not self._is_processed:
            self.process()
            
        if function_name in self._exclusive_times:
            call_chains = self._function_data[function_name]["call_chains"]
            if call_chains:
                most_frequent_chain = max(call_chains.items(), key=lambda x: x[1])[0]
                return FunctionStats(name=function_name, exclusive_time=self._exclusive_times[function_name], call_chain=list(most_frequent_chain))
            else:
                return FunctionStats(name=function_name, exclusive_time=self._exclusive_times[function_name])
        return None


def main():
    parser = argparse.ArgumentParser(description='Analyze folded stacks for exclusive function times')
    parser.add_argument('input_path', 
                       help='Path to directory containing folded stack files or a single .folded file')
    parser.add_argument('-n', '--top_n', type=int, default=10,
                        help='Number of top functions to display (default: 10)')
    parser.add_argument('--show-call-chains', action='store_true',
                        help='Display the call chains for each function')
    parser.add_argument('--function', type=str, default=None,
                        help='Specific function to look up exclusive time for')
    args = parser.parse_args()
    
    analyzer = StackAnalyzer(args.input_path)
    
    if args.function:
        # If a function name is specified, just print that function's exclusive time
        stats = analyzer.get_function_stats(args.function)
        if stats:
            print(f"\nFunction: {stats.name}")
            print(f"Exclusive Time: {stats.exclusive_time:,} samples")
            if args.show_call_chains and stats.call_chain:
                print(f"Call chain: {' -> '.join(stats.call_chain)}")
        else:
            print(f"\nFunction '{args.function}' not found.")
    else:
        # Otherwise, show top N functions
        top_functions = analyzer.get_top_functions(args.top_n)
        
        print(f"\nTop {args.top_n} functions by exclusive time:")
        print("-" * 80)
        for idx, func in enumerate(top_functions, 1):
            print(f"{idx}. {func.name:<40} {func.exclusive_time:>15,} samples")
            if args.show_call_chains and func.call_chain:
                print(f"   Call chain: {' -> '.join(func.call_chain)}")
        print("-" * 80)

if __name__ == "__main__":
    main() 