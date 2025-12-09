#!/usr/bin/env python3
"""
Tests for the LLM client module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.llm_client import (
    LLMClient,
    DummyLLMClient,
    OpenAILLMClient,
    create_llm_client
)


def test_dummy_client_creation():
    """Test DummyLLMClient creation."""
    client = DummyLLMClient()
    
    assert isinstance(client, LLMClient)
    assert client.get_model_name() == "dummy-model"


def test_dummy_client_with_custom_name():
    """Test DummyLLMClient with custom model name."""
    client = DummyLLMClient(model_name="test-model")
    
    assert client.get_model_name() == "test-model"


def test_dummy_client_complete():
    """Test DummyLLMClient completion."""
    client = DummyLLMClient()
    
    prompt = """
    def my_function(n):
        return sum(range(n))
    """
    
    response = client.complete(prompt)
    
    assert isinstance(response, str)
    assert len(response) > 0
    # Should contain some indication it's a placeholder
    assert "dummy" in response.lower() or "placeholder" in response.lower()


def test_dummy_client_extracts_function_name():
    """Test that DummyLLMClient extracts function name from prompt."""
    client = DummyLLMClient()
    
    prompt = """
    def calculate_sum(n):
        return sum(range(n))
    """
    
    response = client.complete(prompt)
    
    # Should mention the function name
    assert "calculate_sum" in response


def test_create_llm_client_dummy():
    """Test factory function for dummy client."""
    client = create_llm_client("dummy")
    
    assert isinstance(client, DummyLLMClient)


def test_create_llm_client_invalid_provider():
    """Test factory function with invalid provider."""
    try:
        client = create_llm_client("invalid_provider")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported provider" in str(e)


def test_openai_client_requires_api_key():
    """Test that OpenAILLMClient requires an API key."""
    # Clear environment variable temporarily
    old_key = os.environ.get('OPENAI_API_KEY')
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    
    try:
        client = OpenAILLMClient()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "API key required" in str(e)
    finally:
        # Restore environment variable
        if old_key:
            os.environ['OPENAI_API_KEY'] = old_key


def test_create_llm_client_with_kwargs():
    """Test factory function passes kwargs correctly."""
    client = create_llm_client("dummy", model_name="custom-model")
    
    assert client.get_model_name() == "custom-model"


def test_dummy_client_deterministic():
    """Test that DummyLLMClient returns consistent results."""
    client = DummyLLMClient()
    
    prompt = "def test(): pass"
    
    response1 = client.complete(prompt)
    response2 = client.complete(prompt)
    
    # Should be deterministic (same input -> same output)
    assert response1 == response2


if __name__ == '__main__':
    # Simple test runner
    import traceback
    
    tests = [
        test_dummy_client_creation,
        test_dummy_client_with_custom_name,
        test_dummy_client_complete,
        test_dummy_client_extracts_function_name,
        test_create_llm_client_dummy,
        test_create_llm_client_invalid_provider,
        test_openai_client_requires_api_key,
        test_create_llm_client_with_kwargs,
        test_dummy_client_deterministic,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
