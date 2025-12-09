#!/usr/bin/env python3
"""
LLM client abstraction for optimization suggestions.

This module provides a flexible interface for interacting with different LLM providers
for code optimization tasks, including a dummy client for offline testing.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Additional provider-specific parameters
        
        Returns:
            The LLM's response as a string
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model being used."""
        pass


class DummyLLMClient(LLMClient):
    """
    Dummy LLM client for offline testing and development.
    
    Returns deterministic placeholder responses without making any API calls.
    Useful for testing the optimization pipeline without network access or API keys.
    """
    
    def __init__(self, model_name: str = "dummy-model"):
        """
        Initialize the dummy client.
        
        Args:
            model_name: Name to report for the model (default: "dummy-model")
        """
        self.model_name = model_name
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a dummy completion.
        
        Returns a placeholder response that mimics the expected format
        of a real LLM optimization suggestion.
        
        Args:
            prompt: The prompt (analyzed but not used for generation)
            **kwargs: Ignored
        
        Returns:
            A deterministic placeholder response
        """
        # Extract the entire original function from the prompt
        import re
        
        # Look for the function in a code block
        code_block_match = re.search(r'```python\n(def\s+\w+.*?)```', prompt, re.DOTALL)
        if code_block_match:
            original_function = code_block_match.group(1).strip()
        else:
            # Try without code blocks
            func_match = re.search(r'(def\s+\w+\s*\([^)]*\):.*?)(?:\n\S|\Z)', prompt, re.DOTALL)
            if func_match:
                original_function = func_match.group(1).strip()
            else:
                # Fallback
                original_function = "def unknown_function():\n    pass"
        
        # Return a properly formatted response that the optimizer can parse
        # For safety, return the original function unchanged but with a comment
        return f"""EXPLANATION:
This is a dummy/placeholder optimization from DummyLLMClient.
In a real scenario, an LLM would analyze the code and suggest specific improvements such as:
- Algorithmic optimizations (e.g., reducing time complexity)
- Data structure improvements (e.g., using sets instead of lists for membership checks)
- Avoiding repeated computations (caching/memoization)
- Using built-in functions or library optimizations
- Generator expressions instead of list comprehensions where appropriate

For testing purposes, this returns the original function unchanged to maintain correctness.
To see real optimizations, use --use-real-llm flag with a valid OpenAI API key.

OPTIMIZED_CODE:
```python
{original_function}
```"""
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name


class OpenAILLMClient(LLMClient):
    """
    OpenAI LLM client for real optimization suggestions.
    
    Requires OpenAI API key to be set via environment variable or constructor.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Model name (default: gpt-3.5-turbo)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 2000)
        
        Raises:
            ValueError: If no API key is provided and OPENAI_API_KEY is not set
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Import OpenAI library (lazy import)
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )
    
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate a completion using OpenAI's API.
        
        Args:
            prompt: The prompt to send to the LLM
            **kwargs: Override default parameters (temperature, max_tokens, etc.)
        
        Returns:
            The LLM's response text
        
        Raises:
            Exception: If the API call fails
        """
        # Override defaults with any provided kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Python performance optimization engineer."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model


def create_llm_client(
    provider: str = "dummy",
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: Either "dummy" for offline testing or "openai" for real API calls
        **kwargs: Provider-specific configuration (api_key, model, etc.)
    
    Returns:
        An instance of LLMClient
    
    Example:
        >>> # For testing without API
        >>> client = create_llm_client("dummy")
        >>> 
        >>> # For production with OpenAI
        >>> client = create_llm_client("openai", api_key="sk-...", model="gpt-4")
    
    Raises:
        ValueError: If provider is not supported
    """
    if provider == "dummy":
        return DummyLLMClient(**kwargs)
    elif provider == "openai":
        return OpenAILLMClient(**kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'dummy' or 'openai'.")
