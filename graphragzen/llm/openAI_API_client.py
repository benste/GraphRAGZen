import os
from typing import Any, List, Optional, Union
import requests
from urllib.parse import urljoin
import json

from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer
from pydantic import BaseModel
import jsonref

from .typing import LlmAPIClientConfig
from graphragzen.llm.base_llm import LLM


class OpenAICompatibleClient(LLM):
    """Uses API enpoints that are compatible with the OpenAI API enpoints"""

    def __init__(self, **kwargs: Union[dict, LlmAPIClientConfig, Any]) -> None:
        self.config = LlmAPIClientConfig(**kwargs)  # type: ignore
        
        self._initiate_tokenizer()
        
        openai_api_key = os.environ.get(self.config.api_key_env_variable)
        if not openai_api_key:
            # Set a fake key that will load the client but won't authenticate to openAI
            openai_api_key = "sk-proj-ZNBiANHU9ilDCsySC0fVeTD76Vc5DNPbstPThgSdtgI0QV51FBBLeNgL2IZVEFfNX548MqAhnAWfJnlEQtzCelsGL73fx8NBuxc3ZTrP8Ux6qgX4c4emqcIrGTnz"
            
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=openai_api_key,
            organization=self.config.openai_organization_id,
            project=self.config.openai_project_id,
            max_retries=self.config.max_retries,
        )

        super().__init__()

    def __call__(
        self, input: Any, **kwargs: Any
    ) -> Any:
        """Call the LLM as you would llm(input) for simple completion.

        Args:
            input (Any): Any input you would normally pass to llm(input, kwargs)
            output_structure (BaseModel, optional): Output structure to force, e.g. grammars from
                llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the reference.
                Correct = BaseLlamCpp("some text", MyPydanticModel)
                Wrong = BaseLlamCpp("some text", MyPydanticModel())
            kwargs (Any): Any keyword arguments you would normally pass to llm(input, kwargs)

        Returns:
            Any
        """
        
        result = self.client.completions.create(
            prompt=input,
            model=self.config.model,
            **kwargs,
        )
        
        # Format and return
        return {"choices": [{"text": result.content}]}
    
    def _initiate_tokenizer(self) -> None:
        """Tries to use the following methods of tokenization in order:
        1. Load tokenizer from HF using self.config.hf_tokenizer_URI
        2. Try to tokenize and de-tokenize using the API endpoints self.config.base_url/tokenize
            and self.config.base_url/detokenize
        3. Try to initiate tiktoken, getting encoding from
            tiktoken.encoding_for_model(self.config.model)
            
        If any tokenizer is success initiated it is set as the class tokenizer and the other methods
        are not tried.
        """
        
        tokenizer_set = False
        
        # Try to load tokenizer from HF
        if self.config.hf_tokenizer_URI:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.hf_tokenizer_URI)
                tokens = self.tokenizer.tokenize("hello world, how are you?")
                detokenized = self.tokenizer.convert_tokens_to_string(tokens)
                tokenizer_set = True
            except Exception:
                tokenizer_set = False
        
        # Try API tokenizer and detokenizer endpoints
        if not tokenizer_set:
            try:
                self.tokenizer = ApiTokenizer(self.config.base_url)
                tokens = self.tokenizer.tokenize("hello world, how are you?")
                detokenized = self.tokenizer.convert_tokens_to_string(tokens)
                tokenizer_set = True
            except Exception:
                tokenizer_set = False
                
        # Try to load and use tiktoken tokenizer
        if not tokenizer_set:
            try:
                self.tokenizer = TikTokenTokenizer(self.config.model)
                tokens = self.tokenizer.tokenize("hello world, how are you?")
                detokenized = self.tokenizer.convert_tokens_to_string(tokens)
                tokenizer_set = True
            except Exception:
                tokenizer_set = False
                
        if not tokenizer_set:
            raise Exception(
                "Failed to load any of huggingface tokenizer, API tokenizer and tiktoken"
                )
            
    def run_chat(
        self,
        chat: List[dict],
        max_tokens: Optional[int] = None,
        output_structure: Optional[BaseModel] = None,
        stream: bool = False,
    ) -> str:
        """Runs a chat through the LLM

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to None
            output_structure (BaseModel, optional): Output structure to force, e.g. grammars from
                llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the reference.
                Correct = llm.run_chat("some text", MyPydanticModel)
                Wrong = llm.run_chat("some text", MyPydanticModel())
            stream (bool, optional): If True, streams the results to console. Defaults to False.

        Returns:
            str: Generated content
        """
        
        cache_results = self.check_cache(str(chat))
        if cache_results: # Check cache first
            results = cache_results
        else: # Use LLM if not in cache
            # Make sure max_tokens is set correctly
            if not max_tokens or max_tokens<0:
                max_tokens = 10**10
                
            if output_structure:
                # Output structure cannot be a BaseModel for llama cpp server
                # Let's convert it to something OpenAI and llama cpp server both understand
                unrefed_schema = jsonref.replace_refs(output_structure.model_json_schema())
                properties = json.loads(json.dumps(unrefed_schema['properties'], indent=2))
                response_format = {
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": properties,
                        "required": output_structure.model_json_schema()['required'],
                        },
                    }
            else:
                response_format = None
                
            results = self.client.chat.completions.create(
                messages=chat,
                model=self.config.model,
                response_format=response_format,
                frequency_penalty=1.0,
                max_tokens=max_tokens,
                stream=stream,
            )
            
            if stream:
                results = self.print_streamed(results)  # type: ignore
            else:
                results = results.choices[0].message.content  # type: ignore

            # And add the result to cache
            self.write_item_to_cache(str(chat), results)

        return results

    def num_chat_tokens(self, chat: List[dict]) -> int:
        """Return the length of the tokenized chat

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

        Returns:
            int: number of tokens
        """
        
        try:
            # If the tokenizer is from HF, we can get the formatted chat as a single string,
            # tokenize it, and get the number of tokens
            return len(self.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True,
                ))
        except Exception:
            # Otherwise we'll join the messages and which will lead to a relatively good estimate
            # of the final prompt string
            formatted_chat = " ".join([
                f"{message['role']} {message['content']}" for message in chat
            ])
            num_tokens = len(self.tokenizer.tokenize(formatted_chat))
            
            # Since it's an estimate, we'd rather overestimate
            return num_tokens + (4 * len(chat))

    def tokenize(self, content: str) -> Union[List[str], List[int]]:
        """Tokenize a string

        Args:
            content (str): String to tokenize

        Returns:
            Union[List[str], List[int]]: Tokenized string
        """

        return self.tokenizer.tokenize(content)

    def untokenize(self, tokens: Union[List[str], List[int]]) -> str:
        """Generate a string from a list of tokens

        Args:
            tokens (Union[List[str], List[int]]): Tokenized string

        Returns:
            str: Untokenized string
        """

        return self.tokenizer.convert_tokens_to_string(tokens)


class ApiTokenizer:
    """tokenizes and detokenizes using API endpoints"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def tokenize(self, content: str, base_url: Optional[str] = None) -> List[int]:
        base_url = base_url or self.base_url
        result = requests.post(urljoin(base_url, 'tokenize'), json = {"content": content})
        return result.json().get("tokens")
    
    def convert_tokens_to_string(self, tokens: List[int], base_url: Optional[str] = None) -> str:
        base_url = base_url or self.base_url
        result = requests.post(urljoin(base_url, 'detokenize'), json = {"tokens": tokens})
        return result.json().get("content")
    
class TikTokenTokenizer:
    """tokenizes and detokenizes using tiktoken library"""
    
    def __init__(self, model_name: str):
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def tokenize(self, content: str) -> List[int]:
        return self.encoding.encode(content)
    
    def convert_tokens_to_string(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)