import os
import logging
import subprocess
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_current_branch(codebase_dir='.'):
    """
    Get the current git branch name
    
    Args:
        codebase_dir: Directory of the git repository
        
    Returns:
        String with current branch name or empty string if error
    """
    try:
        cmd = f'git -C {codebase_dir} rev-parse --abbrev-ref HEAD'
        return os.popen(cmd).read().strip()
    except Exception as e:
        logger.error(f"Error getting current branch: {e}")
        return ""

def checkout_branch(branch_name, codebase_dir='.', quiet=True):
    """
    Checkout a specific git branch
    
    Args:
        branch_name: Name of the branch to checkout
        codebase_dir: Directory of the git repository
        quiet: Whether to suppress git output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        quiet_flag = "-q" if quiet else ""
        cmd = f'git -C {codebase_dir} checkout {branch_name} {quiet_flag}'
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"Error checking out branch {branch_name}: {e}")
        return False

def create_branch(branch_name, codebase_dir='.', quiet=True):
    """
    Create a new git branch
    
    Args:
        branch_name: Name of the new branch
        codebase_dir: Directory of the git repository
        quiet: Whether to suppress git output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        quiet_flag = "-q" if quiet else ""
        cmd = f'git -C {codebase_dir} branch {branch_name} {quiet_flag}'
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"Error creating branch {branch_name}: {e}")
        return False

def reset_hard(codebase_dir='.'):
    """
    Reset git repository to HEAD with --hard flag
    
    Args:
        codebase_dir: Directory of the git repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = f'git -C {codebase_dir} reset --hard HEAD'
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"Error resetting repository: {e}")
        return False

def stage_file(file_path, codebase_dir='.'):
    """
    Stage a file for commit
    
    Args:
        file_path: Path to the file to stage, relative to codebase_dir
        codebase_dir: Directory of the git repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = f'git -C {codebase_dir} add {file_path}'
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"Error staging file {file_path}: {e}")
        return False

def commit_changes(message, codebase_dir='.', quiet=True):
    """Junie
    Commit staged changes
    
    Args:
        message: Commit message
        codebase_dir: Directory of the git repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        quiet_flag = "-q" if quiet else ""
        cmd = f'git -C {codebase_dir} commit -m "{message}" {quiet_flag}'
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"Error committing changes: {e}")
        return False

def push_branch(branch_name, codebase_dir='.', quiet=True):
    """
    Push branch to origin
    
    Args:
        branch_name: Name of the branch to push
        codebase_dir: Directory of the git repository
        
    Returns:
        True if successful, False otherwise
    """
    try:
        quiet_flag = "-q" if quiet else ""
        cmd = f'git {quiet_flag} -C {codebase_dir} push origin {branch_name}'
        result = os.system(cmd)
        return result == 0
    except Exception as e:
        logger.error(f"Error pushing branch {branch_name}: {e}")
        return False


@contextmanager
def temp_checkout(branch_name, codebase_dir='.', quiet=True):
    """
    Context manager for temporarily checking out a branch and returning to the previous branch when done.
    If branch_name is empty or None, no branch switching occurs.
    
    Usage:
        with temp_checkout('feature-branch'):
            # Do work on feature-branch
        # Automatically returns to previous branch after the block
    
    Args:
        branch_name: Name of the branch to checkout
        codebase_dir: Directory of the git repository
        quiet: Whether to suppress git output
        
    Yields:
        None
    """
    # If branch name is empty or None, don't do any checkout
    if not branch_name:
        yield
        return
        
    previous_branch = get_current_branch(codebase_dir)
    logger.debug(f"Previous branch: {previous_branch}, checking out {branch_name}...")
    try:
        # Checkout the requested branch
        if not checkout_branch(branch_name, codebase_dir, quiet):
            logger.error(f"Failed to checkout branch {branch_name}")
            raise RuntimeError(f"Failed to checkout branch {branch_name}")
        
        # Yield control back to the caller
        yield
    finally:
        # Always try to go back to the previous branch
        if previous_branch:
            checkout_branch(previous_branch, codebase_dir, quiet)
            logger.debug(f"Returned to branch {previous_branch}")

class GitBranchFilter(logging.Filter):
    """
    Logging filter that adds the current git branch to log records
    """
    def __init__(self, codebase_dir='.'):
        super().__init__()
        self.codebase_dir = codebase_dir
        
    def filter(self, record):
        branch = get_current_branch(self.codebase_dir)
        record.git_branch = f"\033[38;5;208m[{branch}]\033[0m" if branch else ""
        return True

def setup_branch_logging(codebase_dir='.', log_level=logging.INFO):
    """
    Set up logging with git branch prefix
    
    Args:
        codebase_dir: Directory of the git repository
        log_level: Logging level to use
    """
    root_logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(git_branch)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add git branch filter to root logger
    branch_filter = GitBranchFilter(codebase_dir)
    for handler in logging.getLogger().handlers:
        handler.addFilter(branch_filter)