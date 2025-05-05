import os
from typing import Optional, List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

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
    Supports GPT (OpenAI), Claude (Anthropic), Gemini (Google), LLaMa (Meta), and custom vLLM servers.
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
            provider: The provider of the model (openai, anthropic, google, meta, vllm)
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
            
            elif self.provider == "meta":
                base_url = "https://api.llama.com/compat/v1/"
                return ChatOpenAI(
                    model=self.model,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                    openai_api_key=self.api_key,
                    openai_api_base=base_url,
                    **self.kwargs
                )
                
            elif self.provider == "vllm":
                # Assuming vLLM is running with an OpenAI-compatible API
                base_url = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
                
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
    
    def load_prompt_template(
        self,
        template: str,
        input_variables: List[str],
    ) -> PromptTemplate:
        """
        Create and return a LangChain PromptTemplate.

        Args:
            template: The prompt text with {placeholders}
            input_variables: The variables used in the template

        Returns:
            A PromptTemplate instance
        """
        return PromptTemplate(
            template=template,
            input_variables=input_variables
        )

    def run_chain(
        self,
        prompt_template: PromptTemplate,
        inputs: Dict[str, Any],
        temperature: Optional[float] = None
    ) -> str:
        """
        Run a single prompt template using LLMChain.

        Args:
            prompt_template: A PromptTemplate object
            inputs: A dictionary of input variables for the prompt
            temperature: Optional temperature override

        Returns:
            The LLM's response as a string
        """
        llm = self._initialize_llm(temperature) if temperature else self.llm
        chain = LLMChain(llm=llm, prompt=prompt_template)
        return chain.run(inputs)

    def run_multi_chain(
        self,
        chains: List[LLMChain],
        inputs: Dict[str, Any],
        output_variables: List[str]
    ) -> Dict[str, str]:
        """
        Run a SequentialChain composed of multiple LLMChains.

        Args:
            chains: A list of LLMChain instances
            inputs: The input dictionary
            output_variables: The expected outputs from the final chain

        Returns:
            A dictionary of outputs
        """
        seq = SequentialChain(
            chains=chains,
            input_variables=list(inputs.keys()),
            output_variables=output_variables,
            verbose=True
        )
        return seq.invoke(inputs)
