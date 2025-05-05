from abc import ABC, abstractmethod
from typing import Dict, Tuple, List

from analyzers.clang_remark_analyzer import OptimizationRemark
from analyzers.stack_analyzer import StackAnalyzer


class PerformanceVerifier(ABC):
    """
    Abstract base class for performance verification.
    
    This class defines the interface for performance verification components.
    Subclasses must implement the abstract methods to provide specific
    verification logic for different environments and benchmarks.
    """
    
    def __init__(self, codebase_dir: str):
        """
        Initialize the performance verifier.
        
        Args:
            codebase_dir: Directory containing the codebase to verify
        """
        self.codebase_dir = codebase_dir

    @abstractmethod
    def get_remarks(self, branch="") -> List[OptimizationRemark | str]:
        """
        Retrieve a list of optimization remarks for a given branch.
        
        Args:
            branch: The Git branch to analyze for optimization remarks
                    (empty string defaults to the current branch).
        
        Returns:
            A list of OptimizationRemark objects summarizing compiler
            optimization diagnostics OR a list of yaml content strings of the raw .opt.yaml files.
        """
        pass

    @abstractmethod
    def get_performance(self, branch="", n_iters=1) -> Tuple[Dict, bool]:
        """
        Get performance metrics for a specific branch.
        
        Args:
            branch: Git branch to measure (empty for current branch)
            
        Returns:
            Tuple of (performance_data, tests_pass_flag)
        """
        pass

    @abstractmethod
    def compare_performance(self, performance1, performance2) -> bool:
        """
        Compare two performance measurements to determine if there's an improvement.
        
        Args:
            performance1: Baseline performance data
            performance2: New performance data to compare against baseline
            
        Returns:
            True if performance2 is better than performance1, False otherwise
        """
        pass

    @abstractmethod
    def validate_performance(self, performance) -> bool:
        """
        Validate that performance data is valid and meaningful.
        
        Args:
            performance: Performance data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        pass

    @abstractmethod
    def summarize_improvements(self, performance1, performance2, event="cycles") -> None:
        """
        Print a summary of performance improvements.
        
        Args:
            performance1: Baseline performance data
            performance2: New performance data
            event: Specific metric to highlight in the summary
        """
        pass
        
    @abstractmethod
    def tests_pass(self, branch="") -> bool:
        """
        Check if tests pass on a specific branch.
        
        Args:
            branch: Git branch to test (empty for current branch)
            
        Returns:
            True if tests pass, False otherwise
        """
        pass
        
    @abstractmethod
    def get_compilation_error(self, branch="") -> str:
        """
        Get compilation error messages if any.
        
        Args:
            branch: Git branch to check (empty for current branch)
            
        Returns:
            Error message string if compilation fails, empty string if successful
        """
        pass

    def get_counters_for_function_from_directory(self, directory, function_name):
        analyzer = StackAnalyzer(directory)
        fstats = analyzer.get_function_stats(function_name)
        return fstats.exclusive_time if fstats else -1