from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class PerformanceVerifier(ABC):
    def __init__(self, codebase_dir: str):
        """
        Initializes the PerformanceVerifier class with the directory of a codebase.
        """
        self.codebase_dir = codebase_dir

    @abstractmethod
    def get_performance(self, branch="") -> Tuple[Dict, bool]:
        """
        Retrieves the performance metrics of a given branch.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def compare_performance(self, performance1, performance2) -> bool:
        """
        Compares two performance metrics and returns True if there is an improvement.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def validate_performance(self, performance) -> bool:
        """
        Validates a performance metric.
        This method must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def summarize_improvements(self, performance1, performance2, event="cycles") -> None:
        """
        Summarizes the performance improvements between two performance metrics.
        This method must be implemented by subclasses.
        """
        pass
    @abstractmethod
    def tests_pass(self, branch) -> bool:
        """
        Checks if the tests pass on a given branch.
        This method must be implemented by subclasses.
        """
        pass
        
    @abstractmethod
    def get_compilation_error(self, branch="") -> str:
        """
        Checks if there are compilation errors on a given branch.
        This method must be implemented by subclasses.
        
        Returns:
            Error message string if compilation fails, empty string if successful
        """
        pass