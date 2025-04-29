import os
from typing import Optional, List, Dict, Any

# Import necessary LangChain components
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# Provider-specific imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


class UniversalLLM:
    """
    A universal wrapper for various LLM providers using LangChain.
    Supports OpenAI, Anthropic (Claude), Google Generative AI (Gemini), and custom vLLM.
    """
    
    def __init__(
        self,
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the wrapper with the specified model and provider.
        
        Args:
            model: The name of the model to use
            provider: The provider of the model (openai, anthropic, gemini, vllm)
            temperature: The temperature for generation (default: 0.7)
            max_tokens: The maximum number of tokens to generate (optional)
            **kwargs: Additional arguments to pass to the model
        """
        self.model = model
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Get API key from environment variables
        api_key_env = f"{self.provider.upper()}_API_KEY"
        self.api_key = os.environ.get(api_key_env)
        
        if self.api_key is None:
            raise ValueError(f"API key not found in environment variable {api_key_env}")
        
        # Initialize the appropriate model
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self, temperature_override: Optional[float] = None) -> BaseChatModel:
        """
        Initialize the appropriate LLM based on the provider.
        
        Args:
            temperature_override: Optional temperature to override the default
            
        Returns:
            The initialized LLM
        """
        temp = temperature_override if temperature_override is not None else self.temperature
        
        try:
            if self.provider == "openai":
                params = {
                    "model": self.model,
                    "temperature": temp,
                }
                if self.max_tokens is not None:
                    params["max_tokens"] = self.max_tokens
                
                return ChatOpenAI(**params, **self.kwargs)
                
            elif self.provider == "anthropic":
                params = {
                    "model": self.model,
                    "temperature": temp,
                }
                if self.max_tokens is not None:
                    params["max_tokens"] = self.max_tokens
                
                return ChatAnthropic(**params, **self.kwargs)
                
            elif self.provider == "google":
                params = {
                    "model": self.model,
                    "temperature": temp,
                }
                if self.max_tokens is not None:
                    params["max_output_tokens"] = self.max_tokens
                
                return ChatGoogleGenerativeAI(**params, **self.kwargs)
                
            elif self.provider == "modal":
                # Assuming vLLM is running with an OpenAI-compatible API
                base_url = os.getenv("MODAL_API_BASE", "http://localhost:8000/v1")
                
                params = {
                    "model": self.model,
                    "temperature": temp,
                    "base_url": base_url,
                    "api_key": self.api_key,  # vLLM may not require authentication
                }
                if self.max_tokens is not None:
                    params["max_tokens"] = self.max_tokens
                
                return ChatOpenAI(**params, **{k: v for k, v in self.kwargs.items() if k != "base_url"})
                
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            raise ValueError(f"Error initializing {self.provider} model: {str(e)}")
    
    def _prepare_messages(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Prepare the messages list for the LLM.
        
        Args:
            prompt: The user's prompt
            system_message: Optional system message to guide the model
            context: Optional context to include with the prompt
            
        Returns:
            A list of messages to pass to the model
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        # Add context as part of the prompt if provided
        content = prompt
        if context:
            content = f"{context}\n\n{prompt}"
        
        messages.append(HumanMessage(content=content))
        
        return messages
    
    def prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a response to the given prompt.
        
        Args:
            prompt: The user's prompt
            system_message: Optional system message to guide the model
            context: Optional context to include with the prompt
            temperature: Optional temperature override for this specific prompt
            
        Returns:
            The generated response as a string
        """
        # If temperature is overridden, recreate the LLM
        if temperature is not None and temperature != self.temperature:
            llm = self._initialize_llm(temperature)
        else:
            llm = self.llm
        
        # Prepare messages
        messages = self._prepare_messages(prompt, system_message, context)
        
        try:
            # Generate response
            response = llm.invoke(messages)
            
            # Extract and return the content
            return response.content
        except Exception as e:
            raise ValueError(f"Error generating response: {str(e)}")
    
    async def prompt_async(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Asynchronously generate a response to the given prompt.
        
        Args:
            prompt: The user's prompt
            system_message: Optional system message to guide the model
            context: Optional context to include with the prompt
            temperature: Optional temperature override for this specific prompt
            
        Returns:
            The generated response as a string
        """
        # If temperature is overridden, recreate the LLM
        if temperature is not None and temperature != self.temperature:
            llm = self._initialize_llm(temperature)
        else:
            llm = self.llm
        
        # Prepare messages
        messages = self._prepare_messages(prompt, system_message, context)
        
        try:
            # Generate response asynchronously
            response = await llm.ainvoke(messages)
            
            # Extract and return the content
            return response.content
        except Exception as e:
            raise ValueError(f"Error generating async response: {str(e)}")
