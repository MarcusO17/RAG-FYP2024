from typing import Optional, List, Mapping, Any
import os

from groq import Groq
from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
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
1. Add setters 
2. Add temp
3. Change to JSON mode
"""
class GroqLLM(CustomLLM):
    context_window: int
    num_output: int 
    model_name: str 
    is_chat_model : bool = True
    is_function_calling_model: bool = True
    client: Optional[Groq]
    system_prompt : str = "you are a helpful assistant."


    def __init__(self, **data):
        super().__init__(**data)
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

        )
        return CompletionResponse(text=chat_completion.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        try:
            stream = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "you are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                stream=True,
            )
            
            response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    response += content
                    yield CompletionResponse(text=response, delta=content)
        except Exception as e:
            raise ValueError(f"Error in Groq API streaming: {str(e)}")