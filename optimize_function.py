#!/usr/bin/env python3

import os
import re
import openai
import anthropic
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from extract_deps import extract_dependencies, DependencyExtractorConfig, format_analysis_output, DependencyAnalysis
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
            for func in analysis.functions:
                if func['name'] == function_name:
                    f = func['location'].split(':')[0]
                    if f.endswith('.cpp'):
                        original_file = f
                        original_function = func['body']
                        break
            
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
            # Save current branch name
            current_branch = os.popen('git -C {} rev-parse --abbrev-ref HEAD'.format(codebase_dir)).read().strip()
            
            # Create and checkout new branch
            branch_cmd = f'git -C {codebase_dir} checkout -b {result.branch_name} -q'
            if os.system(branch_cmd) != 0:
                raise ValueError("Failed to create and checkout new branch")
            
            # Read original file
            print(f"Applying optimization to file: {result.file_path}")
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
            stage_cmd = f'git -C {codebase_dir} add {file_relative_path}'
            if os.system(stage_cmd) != 0:
                raise ValueError("Failed to stage changes")
            
            # Commit changes
            commit_cmd = f'git -C {codebase_dir} commit -m "{commit_msg}"'
            if os.system(commit_cmd) != 0:
                raise ValueError("Failed to commit changes")
            
            # push changes
            push_cmd = f'git -C {codebase_dir} push origin {result.branch_name}'
            if os.system(push_cmd) != 0:
                raise ValueError("Failed to push changes")
            
            # Restore original branch
            os.system(f'git -C {codebase_dir} checkout {current_branch} -q')
            
        except Exception as e:
            # Try to restore original branch
            try:
                os.system(f'git -C {codebase_dir} checkout {current_branch} -q')
            except:
                pass
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
    
    args = parser.parse_args()
    
    # Set default model based on provider if not specified by user
    if args.provider == 'anthropic' and args.model == 'gpt-4-turbo-preview':
        args.model = 'claude-3-opus-20240229'
        
    # Get API key from args or environment based on provider
    api_key = None
    if args.provider == 'openai':
        api_key = args.openai_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY or use --openai-key")
            return
    elif args.provider == 'anthropic':
        api_key = args.anthropic_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY or use --anthropic-key")
            return
    
    try:
        if args.provider == 'openai':
            optimizer = FunctionOptimizer(
                openai_api_key=api_key,
                model=args.model,
                provider=args.provider
            )
        elif args.provider == 'anthropic':
            optimizer = FunctionOptimizer(
                anthropic_api_key=api_key,
                model=args.model,
                provider=args.provider
            )
        
        print(f"Analyzing function: {args.function_name}")
        result = optimizer.optimize_function(
            args.codebase_dir,
            args.function_name
        )
        
        print("\nOptimization Summary:")
        print("-" * 60)
        print(result.optimization_summary)
        print("\nCreating branch and applying changes...")
        
        optimizer.apply_optimization(result, args.codebase_dir)
        
        print(f"\nSuccess! Created branch: {result.branch_name}")
        print(f"Modified file: {result.file_path}")
        print("\nReview the changes and merge if satisfied.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main() 