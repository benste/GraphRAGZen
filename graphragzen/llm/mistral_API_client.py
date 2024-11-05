import os
import time
import warnings
from typing import Any, Callable, List, Optional, Union

from graphragzen.llm.base_llm import LLM
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistralai import Mistral
from pydantic._internal._model_construction import ModelMetaclass

from .typing import ChatNames


class MistralClient(LLM):
    """Uses API enpoints that are compatible with the OpenAI API enpoints"""

    chatnames: ChatNames = ChatNames(
        user="user",
        system="system",
        model="assistant",
    )

    def __init__(
        self,
        model_name: str = "mistral-small-latest",
        context_size: int = 32000,
        api_key_env_variable: str = "MISTRAL_API_KEY",
        rate_limit: float = 0.75,
        use_cache: bool = True,
        cache_persistent: bool = True,
        persistent_cache_file: str = "./llm_persistent_cache.yaml",
    ) -> None:
        """
        Args:
            model_name (str, optional): Name of the Mistral model to use.
                Defaults to "mistral-small-latest".
            context_size (int): Context size of the model. Defaults to 128000.
            api_key_env_variable (str): Environment variable to read the Mistral API key from.
                Defaults to "MISTRAL_API_KEY".
            rate_limit (optional, float): Maximum number of calls per second.
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """

        self.context_size = context_size
        self.model_name = model_name
        self.rate_limit = rate_limit
        self.use_cache = use_cache
        self.cache_persistent = cache_persistent
        self.persistent_cache_file = persistent_cache_file

        mistral_api_key = os.environ.get(api_key_env_variable)
        if not mistral_api_key:
            # Need this key
            raise Exception(
                f"Please set your Mistral API key using the env variable {api_key_env_variable}"
            )

        self.client: Mistral = Mistral(api_key=mistral_api_key)

        self.tokenizer = MistralTokenizer.from_model(model_name)

        super().__init__()

    def __call__(
        self,
        input: Any,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        **kwargs: Any,
    ) -> None:
        """MISTRAL API DOES NOT SUPPOR DIRECT COMPLETIONS

        Args:
            input (Any): Ignored
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Ignored
            kwargs (Any): Ignored

        Returns:
            None
        """

        warnings.warn("Mistral API does not support direct completions")
        return None

    def run_chat(
        self,
        chat: List[dict],
        max_tokens: Union[int, None] = -1,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Runs a chat through the LLM

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (Union[int, None], optional): Maximum number of tokens to generate.
                Defaults to -1 (=max).
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Output structure to
                force. e.g. grammars from llama.cpp. When using a pydantic model, only the reference
                should be passed.
                Correct = BaseLlamaCpp("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp("some text", MyPydanticModel())
            stream (bool, optional): If True, streams the results to console. Defaults to False.
            kwargs (Any): Any keyword arguments to add to the lmm call.

        Returns:
            str: Generated content
        """

        cache_results = self.check_cache(chat, input_ischat=True)
        if cache_results:  # Check cache first
            results = cache_results
        else:  # Use LLM if not in cache
            # Make sure we don't exceed the request rate
            self._sleep_if_rate_limited()

            # Make sure max_tokens is set correctly
            if max_tokens is not None and max_tokens < 0:
                max_tokens = None

            # Mistral can only force json, not a specific json structure
            response_format = None
            if output_structure:
                response_format = {"type": "json_object"}

            if stream:
                completion_func: Callable = self.client.chat.stream
            else:
                completion_func = self.client.chat.complete

            response = completion_func(
                model=self.model_name,
                messages=chat,
                response_format=response_format,
                max_tokens=max_tokens,
                **kwargs,
            )

            if stream:
                results = self.print_streamed(response)  # type: ignore
            else:
                results = response.choices[0].message.content  # type: ignore

            # And add the result to cache
            self.write_item_to_cache(chat, results, input_ischat=True)

        return results

    async def a_run_chat(
        self,
        chat: List[dict],
        max_tokens: Union[int, None] = -1,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Runs a chat through the LLM asynchonously

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (Union[int, None], optional): Maximum number of tokens to generate.
                Defaults to -1 (=max).
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Output structure to
                force. e.g. grammars from llama.cpp. When using a pydantic model, only the reference
                should be passed.
                Correct = BaseLlamaCpp("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp("some text", MyPydanticModel())
            stream (bool, optional): Placeholder for compatibility with sync version, not used.
            kwargs (Any): Any keyword arguments to add to the lmm call.

        Returns:
            str: Generated content
        """

        cache_results = self.check_cache(chat, input_ischat=True)
        if cache_results:  # Check cache first
            content = cache_results
        else:  # Use LLM if not in cache
            # Make sure we don't exceed the rate limit
            time.sleep(1 / self.rate_limit)

            # Make sure max_tokens is set correctly
            if max_tokens is not None and max_tokens < 0:
                max_tokens = 10**10

            # Mistral can only force json, not a specific json structure
            if output_structure:
                response_format = {"type": "json_object"}

            response = await self.client.chat.complete_async(
                model=self.model_name,
                messages=chat,  # type: ignore
                response_format=response_format,  # type: ignore
                max_tokens=max_tokens,
                **kwargs,
            )

            results = response.choices[0].message.content  # type: ignore

            # And add the result to cache
            self.write_item_to_cache(chat, results, input_ischat=True)  # type: ignore

        return content

    def num_chat_tokens(self, chat: List[dict]) -> int:
        """Return the length of the tokenized chat

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

        Returns:
            int: number of tokens
        """

        return len(
            self.tokenizer.encode_chat_completion(ChatCompletionRequest(messages=chat)).tokens  # type: ignore  # noqa: E501
        )

    def tokenize(self, content: str) -> Union[List[str], List[int]]:
        """Tokenize a string

        Args:
            content (str): String to tokenize

        Returns:
            Union[List[str], List[int]]: Tokenized string
        """

        tokenized = self.tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=content)])
        )
        return tokenized.tokens

    def untokenize(self, tokens: Union[List[str], List[int]]) -> str:
        """Generate a string from a list of tokens

        Args:
            tokens (Union[List[str], List[int]]): Tokenized string

        Returns:
            str: Untokenized string
        """

        if not tokens:
            return ""

        return self.tokenizer.decode(tokens)
