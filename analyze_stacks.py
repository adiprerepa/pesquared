#!/usr/bin/env python3

import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Iterator
from dataclasses import dataclass
import argparse
import re

@dataclass
class FunctionStats:
    """Statistics for a single function."""
    name: str
    exclusive_time: int

    def __repr__(self) -> str:
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
    
    def __init__(self, input_path: str):
        """
        Initialize the analyzer with either a directory of folded files or a single file.
        
        Args:
            input_path: Path to either a directory containing .folded files or a single .folded file
        """
        self.input_path = input_path
        self._exclusive_times: Dict[str, int] = {}
        self._is_processed = False

    def _is_kernel_or_unknown_function(self, func_name: str) -> bool:
        """Check if the function is a kernel function or unknown."""
        return any(re.search(pattern, func_name) for pattern in self.KERNEL_PATTERNS)

    def _clean_function_name(self, func_name: str) -> str:
        """Clean the function name by removing process prefixes and normalizing."""
        # Remove process prefixes
        for prefix in self.PROCESS_PREFIXES:
            if func_name.startswith(prefix + ';'):
                func_name = func_name[len(prefix) + 1:]
                break
        
        # Remove any remaining [unknown] parts from the path but keep function name
        parts = func_name.split(';')
        if parts:
            # Keep the last part (the actual function name)
            func_name = parts[-1].strip()
            
        return func_name

    def _parse_folded_stack(self, line: str) -> Tuple[List[str], int]:
        """
        Parse a single line of folded stack into stack frames and sample count.
        Format: path1;path2;...;leaf_function sample_count
        Example: main.exe;_start;[unknown];main;setFrame;refresh;calcColor 48309494
        """
        try:
            # Split into stack trace and count
            stack_str, count_str = line.strip().rsplit(None, 1)
            count = int(count_str)
            
            # Split stack into frames and clean each frame
            frames = [frame.strip() for frame in stack_str.split(';')]
            
            # Remove process name prefix (e.g., main.exe)
            if frames and frames[0] in self.PROCESS_PREFIXES:
                frames = frames[1:]
            
            # Remove [unknown] frames
            frames = [f for f in frames if f != '[unknown]' and f.strip()]
            
            return frames, count
        except (ValueError, IndexError):
            return [], 0

    def _process_file(self, filepath: str) -> Dict[str, int]:
        """Process a single folded stack file."""
        exclusive_times = defaultdict(int)
        
        with open(filepath, 'r') as f:
            for line in f:
                frames, count = self._parse_folded_stack(line)
                if not frames:
                    continue
                
                # Get the leaf function (last frame)
                leaf_function = frames[-1]
                
                # Only exclude if it matches kernel patterns
                if not self._is_kernel_or_unknown_function(leaf_function):
                    exclusive_times[leaf_function] += count
        
        return exclusive_times

    def process(self) -> None:
        """Process all stack files and compute exclusive times."""
        if self._is_processed:
            return

        all_exclusive_times = defaultdict(int)
        
        if os.path.isfile(self.input_path):
            # Single file mode
            file_times = self._process_file(self.input_path)
            all_exclusive_times.update(file_times)
        else:
            # Directory mode
            for filename in os.listdir(self.input_path):
                if not filename.endswith('.folded'):
                    continue
                
                filepath = os.path.join(self.input_path, filename)
                file_times = self._process_file(filepath)
                
                # Aggregate times across files
                for func, time in file_times.items():
                    all_exclusive_times[func] += time
        
        self._exclusive_times = all_exclusive_times
        self._is_processed = True

    def get_top_functions(self, n: int = 10) -> List[FunctionStats]:
        """
        Get the top N functions by exclusive time.
        
        Args:
            n: Number of top functions to return
            
        Returns:
            List of FunctionStats objects for the top N functions
        """
        if not self._is_processed:
            self.process()
            
        sorted_funcs = sorted(self._exclusive_times.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        return [FunctionStats(name=name, exclusive_time=time) 
                for name, time in sorted_funcs[:n]]

    def iter_functions(self) -> Iterator[FunctionStats]:
        """
        Iterate over all functions in order of exclusive time.
        
        Returns:
            Iterator yielding FunctionStats objects
        """
        if not self._is_processed:
            self.process()
            
        sorted_funcs = sorted(self._exclusive_times.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        return (FunctionStats(name=name, exclusive_time=time) 
                for name, time in sorted_funcs)

    def get_function_stats(self, function_name: str) -> Union[FunctionStats, None]:
        """
        Get stats for a specific function.
        
        Args:
            function_name: Name of the function to get stats for
            
        Returns:
            FunctionStats object if function exists, None otherwise
        """
        if not self._is_processed:
            self.process()
            
        if function_name in self._exclusive_times:
            return FunctionStats(
                name=function_name,
                exclusive_time=self._exclusive_times[function_name]
            )
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze folded stacks for exclusive function times')
    parser.add_argument('input_path', 
                       help='Path to directory containing folded stack files or a single .folded file')
    parser.add_argument('-n', '--top_n', type=int, default=10,
                        help='Number of top functions to display (default: 10)')
    args = parser.parse_args()
    
    analyzer = StackAnalyzer(args.input_path)
    top_functions = analyzer.get_top_functions(args.top_n)
    
    # Print results
    print(f"\nTop {args.top_n} functions by exclusive time:")
    print("-" * 60)
    for idx, func in enumerate(top_functions, 1):
        print(f"{idx}. {func.name:<40} {func.exclusive_time:>15,} samples")
    print("-" * 60)

if __name__ == "__main__":
    main() 