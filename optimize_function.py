#!/usr/bin/env python3

import os
import re
import openai
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
                 openai_api_key: str,
                 model: str = "gpt-4-turbo-preview"):
        """
        Initialize the optimizer.
        
        Args:
            openai_api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4-turbo-preview)
        """
        self.model = model
        self.client = openai.OpenAI(api_key=openai_api_key)

    def _create_optimization_prompt(self, 
                                  function_name: str, 
                                  analysis_output: str) -> str:
        """Create the prompt for the LLM optimization."""
        return f"""You are an expert C++ performance optimization engineer. You will be given a detailed analysis of a C++ function and its dependencies. Your task is to optimize this function for better performance.

The analysis includes:
1. The original function's implementation
2. All type definitions it depends on
3. All function calls it makes
4. The complete context of how the function is used

YOU CANNOT CHANGE THE SIGNATURE OR FUNCTIONALITY OF THE FUNCTION. If there are no fruitful otimizations, you can return the original function as is.

Please provide:
1. An optimized version of the function that is a drop-in replacement with THE SAME SIGNATURE AND BEHAVIOR (or the original function if no optimizations are possible)
2. A brief explanation of the optimizations made
3. A short, git-branch-name-friendly summary of the main optimization (e.g. "vectorize-matrix-multiply" or "reduce-memory-allocations")

Please format your response as follows:
OPTIMIZATION_SUMMARY: <one-line summary suitable for a git branch name>
EXPLANATION: <brief explanation of optimizations>
OPTIMIZED_FUNCTION:
<optimized function code>

The optimized function must:
1. Be a drop-in replacement (same signature and behavior)
2. Maintain the same external behavior and correctness
3. Do not use any formatting in your response, what follows from OPTIMIZED_FUNCTION needs to be just the drop-in replacement code.
4. Nothing else should be included in OPTIMIZED_FUNCTION except the optimized function code.

Function to optimize: {function_name}

Analysis:
{analysis_output}

"""

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
        function_match = re.search(r'OPTIMIZED_FUNCTION:\s*\n(.+?)(?=\Z)', 
                                response, re.DOTALL)
        if function_match:
            function_code = function_match.group(1).strip()
            # Remove triple backticks and optional language specifier
            function_code = re.sub(r'```[a-zA-Z]*\n?', '', function_code).strip()
            parts['function'] = function_code
            
        return parts

    def optimize_function(self, 
                         codebase_dir: str,
                         function_name: str,
                         analysis: DependencyAnalysis, optimized_count=0) -> OptimizationResult:
        """
        Optimize a function using LLM and create a git branch with the changes.
        
        Args:
            codebase_dir: Directory containing the C++ codebase
            function_name: Name of the function to optimize
            
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
            # analysis = extract_dependencies(codebase_dir, function_name, config)
            analysis_output = format_analysis_output(
                analysis.functions,
                analysis.types,
                config
            )
            
            # Create optimization prompt
            prompt = self._create_optimization_prompt(function_name, analysis_output)
            print(f"Prompt: {prompt}")
            # return
            # Get optimization from LLM using new OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert C++ performance optimization engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse LLM response
            result = self._parse_llm_response(response.choices[0].message.content)
            if not all(k in result for k in ['summary', 'explanation', 'function']):
                raise ValueError("Invalid LLM response format")
            
            # Get original function location
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
            
            # Create optimization result
            return OptimizationResult(
                original_function=original_function.strip(),
                optimized_function=result['function'].strip(),
                optimization_summary=result['explanation'].strip(),
                branch_name=self._sanitize_branch_name(result['summary']),
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
            branch_cmd = f'git -C {codebase_dir} checkout -b {result.branch_name}'
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
            os.system(f'git -C {codebase_dir} checkout {current_branch}')
            
        except Exception as e:
            # Try to restore original branch
            try:
                os.system(f'git -C {codebase_dir} checkout {current_branch}')
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
    parser.add_argument('--model',
                       default='gpt-4-turbo-preview',
                       help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.openai_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY or use --openai-key")
        return
    
    try:
        optimizer = FunctionOptimizer(
            openai_api_key=api_key,
            model=args.model
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