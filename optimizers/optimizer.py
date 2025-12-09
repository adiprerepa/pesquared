#!/usr/bin/env python3
"""
Python function optimizer using LLM guidance.

This module orchestrates the optimization pipeline:
profile → construct prompt → query LLM → return optimization result.
"""

import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Any, Optional

from analyzers.profiler import ProfileSummary, profile_function, format_profile_summary
from optimizers.llm_client import LLMClient, create_llm_client


@dataclass
class OptimizationResult:
    """Result of an LLM-guided optimization."""
    original_code: str
    optimized_code: str
    profile_summary: ProfileSummary
    explanation: str
    model_used: str


def construct_optimization_prompt(
    function_code: str,
    profile_summary: ProfileSummary,
    function_name: str
) -> str:
    """
    Construct an LLM prompt for code optimization.
    
    Args:
        function_code: Source code of the function to optimize
        profile_summary: Profiling results showing performance bottlenecks
        function_name: Name of the function being optimized
    
    Returns:
        A structured prompt for the LLM
    """
    profile_text = format_profile_summary(profile_summary, llm_friendly=True)
    
    prompt = f"""You are an expert Python performance optimization engineer.

Your task is to optimize the following Python function for better performance.

## Performance Profile
{profile_text}

## Original Function Code
```python
{function_code}
```

## Instructions
1. Analyze the function and the profiling data to identify performance bottlenecks
2. Suggest specific optimizations such as:
   - Better algorithms or data structures
   - Caching/memoization
   - Avoiding repeated computations
   - Using built-in functions or libraries
   - List comprehensions → generators where appropriate
   - Other Python-specific optimizations
3. **CRITICAL**: Your optimized code MUST preserve the exact behavior and function signature
4. Output ONLY the optimized function code, no additional explanations or text
5. The code should be production-ready and maintain correctness

## Output Format
Provide your response in this exact format:

EXPLANATION:
<Brief explanation of the optimizations you made>

OPTIMIZED_CODE:
```python
<Your optimized function code here>
```

Function to optimize: {function_name}
"""
    
    return prompt


def parse_llm_response(response: str) -> tuple[str, str]:
    """
    Parse LLM response to extract explanation and optimized code.
    
    Args:
        response: Raw LLM response text
    
    Returns:
        Tuple of (explanation, optimized_code)
    
    Raises:
        ValueError: If response cannot be parsed
    """
    import re
    
    # Extract explanation
    explanation_match = re.search(
        r'EXPLANATION:\s*\n(.*?)(?=OPTIMIZED_CODE:|$)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    # Extract optimized code (handle both with and without code blocks)
    code_match = re.search(
        r'OPTIMIZED_CODE:\s*\n(?:```python\n)?(.*?)(?:```|$)',
        response,
        re.DOTALL | re.IGNORECASE
    )
    
    if not code_match:
        # Try alternative format
        code_match = re.search(
            r'```python\n(.*?)```',
            response,
            re.DOTALL
        )
    
    if not code_match:
        raise ValueError("Could not extract optimized code from LLM response")
    
    optimized_code = code_match.group(1).strip()
    
    return explanation, optimized_code


def optimize_with_llm(
    func: Callable,
    *args,
    llm_client: Optional[LLMClient] = None,
    n_runs: int = 10,
    **kwargs
) -> OptimizationResult:
    """
    Optimize a function using LLM guidance.
    
    This is the main orchestration function that:
    1. Profiles the function
    2. Constructs an optimization prompt
    3. Queries the LLM
    4. Returns the optimization result
    
    Args:
        func: The function to optimize
        *args: Arguments to pass when profiling the function
        llm_client: LLM client to use (if None, creates a DummyLLMClient)
        n_runs: Number of profiling runs (default: 10)
        **kwargs: Keyword arguments to pass when profiling the function
    
    Returns:
        OptimizationResult with original code, optimized code, and metadata
    
    Example:
        >>> def slow_sum(n):
        ...     result = 0
        ...     for i in range(n):
        ...         result += i
        ...     return result
        >>> 
        >>> # Use dummy client for testing
        >>> result = optimize_with_llm(slow_sum, 10000, n_runs=5)
        >>> print(result.explanation)
        >>> print(result.optimized_code)
        >>> 
        >>> # Use real OpenAI client
        >>> from optimizers.llm_client import create_llm_client
        >>> client = create_llm_client("openai", model="gpt-4")
        >>> result = optimize_with_llm(slow_sum, 10000, llm_client=client)
    """
    # Use dummy client if none provided
    if llm_client is None:
        llm_client = create_llm_client("dummy")
    
    # Step 1: Profile the function
    profile_summary = profile_function(func, *args, n_runs=n_runs, **kwargs)
    
    # Step 2: Get the function source code
    try:
        source_code = inspect.getsource(func)
        # Dedent to remove leading whitespace
        source_code = textwrap.dedent(source_code)
    except (TypeError, OSError) as e:
        raise ValueError(f"Could not extract source code for function: {e}")
    
    function_name = func.__name__
    
    # Step 3: Construct prompt
    prompt = construct_optimization_prompt(source_code, profile_summary, function_name)
    
    # Step 4: Query LLM
    llm_response = llm_client.complete(prompt)
    
    # Step 5: Parse response
    explanation, optimized_code = parse_llm_response(llm_response)
    
    # Step 6: Return result
    return OptimizationResult(
        original_code=source_code,
        optimized_code=optimized_code,
        profile_summary=profile_summary,
        explanation=explanation,
        model_used=llm_client.get_model_name()
    )
