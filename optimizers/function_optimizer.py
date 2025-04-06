#!/usr/bin/env python3

import os
import re
import openai
import anthropic
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from analyzers.dependency_extractor import extract_dependencies, DependencyExtractorConfig, format_analysis_output, DependencyAnalysis
from utils import git_utils

logger = logging.getLogger(__name__)

# Token tracker for pricing calculation
class TokenTracker:
    """Track token usage and calculate costs."""
    
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.input_cost_per_mtok = 3  # $3 per million tokens
        self.output_cost_per_mtok = 15  # $15 per million tokens
        
    def add_usage(self, input_tokens: int, output_tokens: int):
        """Add token usage to the tracker."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        
    def get_cost(self):
        """Calculate cost based on token usage."""
        input_cost = (self.input_tokens / 1_000_000) * self.input_cost_per_mtok
        output_cost = (self.output_tokens / 1_000_000) * self.output_cost_per_mtok
        return input_cost, output_cost, input_cost + output_cost
    
    def __str__(self):
        input_cost, output_cost, total_cost = self.get_cost()
        return f"\033[38;5;28m💲 Token usage: {self.input_tokens:,} input, {self.output_tokens:,} output | Cost: ${total_cost:.4f} (Input: ${input_cost:.4f}, Output: ${output_cost:.4f})\033[0m"

# Global token tracker
token_tracker = TokenTracker()
@dataclass
class OptimizationResult:
    """Result of an LLM-based function optimization."""
    original_function: str
    optimized_function: str
    optimization_summary: str
    branch_name: str
    file_path: str

    def __str__(self) -> str:
        return f"OptimizationResult(summary='{self.optimization_summary}', branch='{self.branch_name}, file='{self.file_path}')"

class FunctionOptimizer:
    """Handles LLM-based function optimization and git operations."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4-turbo-preview",
                 provider: str = "openai"):
        """
        Initialize the optimizer.
        
        Args:
            openai_api_key: OpenAI API key (required if provider is 'openai')
            anthropic_api_key: Anthropic API key (required if provider is 'anthropic')
            model: Model to use for optimization (default depends on provider)
            provider: The LLM provider to use ('openai' or 'anthropic')
        """
        self.model = model
        self.provider = provider
        
        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required when provider is 'openai'")
            self.client = openai.OpenAI(api_key=api_key)
        elif provider == "anthropic":
            if not api_key:
                raise ValueError("Anthropic API key is required when provider is 'anthropic'")
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_optimization_prompt(self, 
                                  function_name: str, 
                                  analysis_output: str,
                                  compilation_error: str = None,
                                  original_function: str = None,
                                  is_parent_in_chain: bool = False,
                                  child_functions: list = None) -> str:
        """
        Create the prompt for the LLM optimization.
        
        Args:
            function_name: Name of the function to optimize
            analysis_output: Analysis of the function and its dependencies
            compilation_error: Optional compilation error from a previous attempt
            original_function: Original unmodified function for re-attempts
            is_parent_in_chain: Whether this function is a parent in a call chain
            child_functions: List of child function names in the call chain
        """
        base_prompt = f"""You are an expert C++ performance optimization engineer tasked with optimizing a function for better performance.

CRITICAL REQUIREMENTS:
1. DO NOT modify the function signature (parameter types, return type, name) - it MUST remain EXACTLY as provided
2. DO NOT inline the function or merge it with others
3. ONLY modify the internal implementation while preserving exact behavior
4. If no valuable optimizations exist, return the original function unmodified
5. Your code MUST compile without errors
"""

        # Add call chain specific instructions if this is a parent function
        if is_parent_in_chain and child_functions:
            children_list = ", ".join(child_functions)
            base_prompt += f"""
IMPORTANT CALL CHAIN CONTEXT:
This function is a parent in a call chain that includes the following child functions: {children_list}.
Your primary optimization goal is to REDUCE THE NUMBER OF CALLS to these child functions by:
- Implementing logic that avoids unnecessary calls to child functions
- Batching or combining multiple calls to the same child function when possible
- Ensuring that child functions are only invoked when truly needed
- Caching results from child function calls where appropriate
- Restructuring logic to eliminate redundant child function calls
- Moving expensive calculations to avoid repeated child function calls

Focus on minimizing call frequency to child functions as your top priority.
"""

        base_prompt += f"""
The provided analysis includes:
- Original function implementation
- All dependent type definitions
- All function calls made
- Complete usage context

Format your response exactly as:
OPTIMIZATION_SUMMARY: <brief git-branch-friendly description>
EXPLANATION: <concise explanation of optimizations>
OPTIMIZED_FUNCTION:
<optimized code - no formatting, just the raw function code, no explanation after, surround with ```>

Function to optimize: {function_name}

Analysis:
{analysis_output}
"""

        # Add compilation error context if provided
        if compilation_error and original_function:
            error_context = f"""
IMPORTANT: Your previous optimization attempt resulted in compilation errors.

Original function:
```
{original_function}
```

Compilation error:
```
{compilation_error}
```

Please fix these errors in your new implementation while maintaining optimizations.
"""
            return base_prompt + error_context
        
        return base_prompt

    def _sanitize_branch_name(self, name: str, optimized_count=0) -> str:
        """Convert an optimization summary into a valid git branch name."""
        # Replace spaces and special characters with hyphens
        name = re.sub(r'[^a-zA-Z0-9-]', '-', name.lower())
        # Remove consecutive hyphens
        name = re.sub(r'-+', '-', name)
        # Remove leading/trailing hyphens
        return f"ai-{optimized_count}/{name.strip('-')}"

    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response into components."""
        parts = {}
        
        # Extract optimization summary
        summary_match = re.search(r'OPTIMIZATION_SUMMARY:\s*(.+?)(?:\n|$)', response)
        if summary_match:
            parts['summary'] = summary_match.group(1).strip()
            
        # Extract explanation
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=\nOPTIMIZED_FUNCTION:|\Z)', 
                                    response, re.DOTALL)
        if explanation_match:
            parts['explanation'] = explanation_match.group(1).strip()
            
        # Extract optimized function 
        function_match = re.search(r'OPTIMIZED_FUNCTION:\s*\n(?:```(?:cpp)?\n)?(.*?)(?:```|\Z)', 
                                response, re.DOTALL)
        if function_match:
            parts['function'] = function_match.group(1).strip()
            
        return parts

    def optimize_function(self, 
                         codebase_dir: str,
                         function_name: str,
                         analysis: DependencyAnalysis, 
                         optimized_count=0,
                         compilation_error: str = None,
                         previous_result: OptimizationResult = None,
                         temperature=0.7,
                         call_chain: list = None,
                         position_in_chain: int = None) -> OptimizationResult:
        """
        Optimize a function using LLM and create a git branch with the changes.
        
        Args:
            codebase_dir: Directory containing the C++ codebase
            function_name: Name of the function to optimize
            analysis: Dependency analysis of the function
            optimized_count: Count of functions optimized so far
            compilation_error: Optional compilation error from a previous attempt
            previous_result: Previous optimization result if this is a retry
            call_chain: The complete call chain this function is part of, from parent to child
            position_in_chain: The position of this function in the call chain (0-indexed)
            
        Returns:
            OptimizationResult containing the optimization details
            
        Raises:
            ValueError: If function not found or optimization fails
        """
        # Extract function dependencies
        config = DependencyExtractorConfig(
            include_function_locations=True,
            include_type_locations=True
        )
        
        try:
            # Get function analysis
            analysis_output = format_analysis_output(
                analysis.functions,
                analysis.types,
                config
            )
            
            # Get original function location and body
            original_file = None
            original_function = None
            cpp_file = None
            cpp_function = None
            header_file = None
            header_function = None
            logger.debug(f"found functions: {analysis.functions}")

            # Define function to sanitize function names (remove templating)
            def sanitize_function_name(name):
                # Remove template parameters and ::, e.g., stack<20>::push -> push 
                return re.sub(r'<.*?>', '', name).split('::')[-1]
            
            # Sanitize the target function name
            sanitized_target = sanitize_function_name(function_name)
            
            for func in analysis.functions:
                # Sanitize the current function name for comparison
                sanitized_func_name = sanitize_function_name(func['name'])
                logger.debug(f"sanitized_func_name: {sanitized_func_name}, sanitized_target: {sanitized_target}, func: {func}")
                
                if sanitized_func_name == sanitized_target:
                    f = func['location'].split(':')[0]
                    # Prioritize implementation files (.cpp, .cc, .cxx)
                    if f.endswith(('.cpp', '.cc', '.cxx')):
                        cpp_file = f
                        cpp_function = func['body']
                        logger.debug(f"cpp_file: {cpp_file}, cpp_function: {cpp_function}")
                        # If we find an implementation file with function body, use it immediately
                        if cpp_function and len(cpp_function.strip()) > 0 and '{' in cpp_function:
                            original_file = cpp_file
                            original_function = cpp_function
                            break
                    # Otherwise capture header files (.h, .hpp, .hxx)
                    elif f.endswith(('.h', '.hpp', '.hxx')):
                        header_file = f
                        header_function = func['body']
                        # If we find a header with function body and not just declaration
                        if header_function and len(header_function.strip()) > 0 and '{' in header_function:
                            # Store but continue looking for cpp implementation
                            if not original_file:
                                original_file = header_file
                                original_function = header_function
            
            # If we couldn't find any implementation, but have a cpp location, use it
            if (not original_file or not original_function) and cpp_file:
                original_file = cpp_file
                original_function = cpp_function
            
            # If we still couldn't find any implementation, but have a header with body, use it
            if (not original_file or not original_function) and header_file:
                original_file = header_file
                original_function = header_function
            
            if not original_file or not original_function:
                raise ValueError(f"Could not locate original function: {function_name}")
            
            # Determine if this is a parent function in a call chain and which child functions it calls
            is_parent_in_chain = False
            child_functions = None
            
            if call_chain and position_in_chain is not None and position_in_chain < len(call_chain) - 1:
                is_parent_in_chain = True
                # Get all child functions in the call chain that come after this function
                child_functions = call_chain[position_in_chain + 1:]
            
            # Create optimization prompt, with error context if this is a retry
            if compilation_error and previous_result:
                prompt = self._create_optimization_prompt(
                    function_name, 
                    analysis_output,
                    compilation_error,
                    previous_result.original_function,
                    is_parent_in_chain,
                    child_functions
                )
            else:
                prompt = self._create_optimization_prompt(
                    function_name, 
                    analysis_output,
                    is_parent_in_chain=is_parent_in_chain,
                    child_functions=child_functions
                )
            
            # Get optimization from LLM based on the selected provider
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert C++ performance optimization engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4000
                )
                response_text = response.choices[0].message.content
                
                # Track token usage
                token_tracker.add_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    system="You are an expert C++ performance optimization engineer.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4000
                )
                response_text = response.content[0].text
                
                # Track token usage
                token_tracker.add_usage(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse LLM response
            result = self._parse_llm_response(response_text)
            if not all(k in result for k in ['summary', 'explanation', 'function']):
                raise ValueError("Invalid LLM response format")
            
            # Create optimization result - use original branch name if this is a retry
            branch_name = previous_result.branch_name if previous_result else self._sanitize_branch_name(result['summary'], optimized_count=optimized_count)
            
            return OptimizationResult(
                original_function=original_function.strip(),
                optimized_function=result['function'].strip(),
                optimization_summary=result['explanation'].strip(),
                branch_name=branch_name,
                file_path=original_file
            )
            
        except Exception as e:
            raise ValueError(f"Optimization failed: {str(e)}")

    def apply_optimization(self, result: OptimizationResult, codebase_dir: str) -> None:
        """
        Apply an optimization by creating a new branch and committing the changes.
        
        Args:
            result: OptimizationResult from optimize_function()
            codebase_dir: Directory containing the C++ codebase
            
        Raises:
            ValueError: If git operations fail
        """
        try:
            # Create and checkout new branch
            if not git_utils.create_branch(result.branch_name, codebase_dir):
                raise ValueError("Failed to create and checkout new branch")
            
            # Use temp_checkout context manager to automatically return to the original branch
            with git_utils.temp_checkout(result.branch_name, codebase_dir):
                # Read original file
                logger.info(f"Applying optimization to file: {result.file_path}")
                with open(os.path.join(codebase_dir, result.file_path), 'r') as f:
                    content = f.read()
                
                # Replace function implementation
                # Escape special regex characters in the original function
                escaped_original = re.escape(result.original_function)
                new_content = re.sub(
                    escaped_original,
                    result.optimized_function,
                    content
                )
                
                # Write changes
                with open(os.path.join(codebase_dir, result.file_path), 'w') as f:
                    f.write(new_content)
                
                # Stage and commit changes
                file_relative_path = os.path.relpath(result.file_path, codebase_dir)
                commit_msg = f"perf: {result.optimization_summary}\n\nOptimized {os.path.basename(result.file_path)}"
                
                # Stage changes
                if not git_utils.stage_file(file_relative_path, codebase_dir):
                    raise ValueError("Failed to stage changes")
                
                # Commit changes
                if not git_utils.commit_changes(commit_msg, codebase_dir, quiet=logger.level == logging.INFO):
                    raise ValueError("Failed to commit changes")
                
                # push changes
                if not git_utils.push_branch(result.branch_name, codebase_dir, quiet=logger.level == logging.INFO):
                    raise ValueError("Failed to push changes")
            
        except Exception as e:
            raise ValueError(f"Failed to apply optimization: {str(e)}")

def main():
    """Command-line interface for the function optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize C++ functions using LLM'
    )
    parser.add_argument('codebase_dir',
                       help='Directory containing the C++ codebase')
    parser.add_argument('function_name',
                       help='Name of the function to optimize')
    parser.add_argument('--openai-key',
                       help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--anthropic-key',
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--provider', 
                       choices=['openai', 'anthropic'], 
                       default='openai',
                       help='LLM provider to use (default: openai)')
    parser.add_argument('--model',
                       default='gpt-4-turbo-preview',
                       help='Model to use (defaults depend on provider)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging with git branch prefix
    codebase_dir = os.path.abspath(args.codebase_dir)
    level = logging.DEBUG if args.debug else logging.INFO
    git_utils.setup_branch_logging(codebase_dir, level)
    
    # Set default model based on provider if not specified by user
    if args.provider == 'anthropic' and args.model == 'gpt-4-turbo-preview':
        args.model = 'claude-3-opus-20240229'
        
    # Get API key from args or environment based on provider
    api_key = None
    if args.provider == 'openai':
        api_key = args.openai_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("🔑 Error: OpenAI API key required. Set OPENAI_API_KEY or use --openai-key")
            return
    elif args.provider == 'anthropic':
        api_key = args.anthropic_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.error("🔑 Error: Anthropic API key required. Set ANTHROPIC_API_KEY or use --anthropic-key")
            return
    
    try:
        optimizer = FunctionOptimizer(
            api_key=api_key,
            model=args.model,
            provider=args.provider
        )
        
        logger.info(f"🔍 Analyzing function: {args.function_name}")
        result = optimizer.optimize_function(
            args.codebase_dir,
            args.function_name
        )
        
        logger.info("\n✨ Optimization Summary:")
        logger.info("-" * 60)
        logger.info(result.optimization_summary)
        logger.info("\n🔧 Creating branch and applying changes...")
        
        optimizer.apply_optimization(result, args.codebase_dir)
        
        logger.info(f"\n✅ Success! Created branch: {result.branch_name}")
        logger.info(f"📝 Modified file: {result.file_path}")
        logger.info("\n👀 Review the changes and merge if satisfied.")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")

if __name__ == '__main__':
    main() 