from abc import ABC, abstractmethod

class PerformanceVerifier(ABC):
    def __init__(self, codebase_dir: str):
        """
        Initializes the PerformanceVerifier class with the directory of a codebase.
        """
        self.codebase_dir = codebase_dir

    @abstractmethod
    def get_performance(self, branch=""):
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