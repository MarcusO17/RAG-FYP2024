from typing import Optional, List, Mapping, Any
import os

from groq import Groq
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings

""" 
TODO
1. Add setters (REDACTED)
2. Add temp and params (DONE)
3. Change to JSON mode (REDACTED)
3.5 Change to Full Pydantic (REFACTORED, UNABLE TO CHANGE)
4. Fix stream (DONE)
4.5 let client be outside this class
5. Add error validation and catching
"""

class GroqLLM(CustomLLM):
    context_window: int 
    num_output: int 
    model_name: str 
    is_chat_model : bool = True
    is_function_calling_model: bool = True
    client: Optional[Groq]
    #Only recommended to change either one but NOT BOTH                  
    top_p: float = 1.0
    temperature: float = 1.0
    tools: Optional[Any]
    system_prompt : str = "you are a helpful assistant."

    def __init__(self, **params):
        super().__init__(**params)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY")) 

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
            is_function_calling_model=self.is_function_calling_model
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature= self.temperature,
            top_p= self.top_p,
            tools= None,
        )
        return CompletionResponse(text=chat_completion.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        stream = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature= self.temperature,
            top_p= self.top_p,
            tools= None,
            stream=True,
        )
        
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                response += content
                yield CompletionResponse(text=response, delta=content)
