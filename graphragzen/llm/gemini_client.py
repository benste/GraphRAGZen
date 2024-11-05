import asyncio
import json
import os
import time
import warnings
from typing import Any, Iterator, List, Optional, Union
from urllib.parse import urljoin

import jsonref
import requests
from graphragzen.llm.base_llm import LLM
from graphragzen.llm.custom_tokenizers import TikTokenTokenizer
from pydantic._internal._model_construction import ModelMetaclass

from .typing import ChatNames


class GeminiClient(LLM):
    """Call the Gemini API"""

    chatnames: ChatNames = ChatNames(
        user="user",
        system="system",
        model="model",
    )

    def __init__(
        self,
        base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/models/",
        model_name: str = "gemini-1.5-flash-latest",
        context_size: int = 10**6,
        api_key_env_variable: str = "GEMINI_API_KEY",
        max_retries: int = 3,
        rate_limit: float = 0.25,
        use_cache: bool = True,
        cache_persistent: bool = True,
        persistent_cache_file: str = "./llm_persistent_cache.yaml",
    ) -> None:
        """
        Args:
            base_url (str, optional): Defaults to
                "https://generativelanguage.googleapis.com/v1beta/models/".
            model_name (str, optional): Name of the model to use, see
                https://ai.google.dev/gemini-api/docs/models/gemini.
                Defaults to "gemini-1.5-flash-latest".
            context_size (int): Context size of the model. Defaults to 10**6.
            api_key_env_variable (str): Environment variable to read the Gemini API key from.
                Defaults to "GEMINI_API_KEY".
            max_retries (optional, int): Number of times to retry on timeout. Defaults to 3.
            rate_limit (optional, float): Maximum number of calls per second.
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """

        self.url = base_url
        self.context_size = context_size
        self.model_name = model_name
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.use_cache = use_cache
        self.cache_persistent = cache_persistent
        self.persistent_cache_file = persistent_cache_file

        self.gemini_api_key = os.environ.get(api_key_env_variable)

        self.tokenizer = TikTokenTokenizer("cl100k_base")

        super().__init__()

    def __call__(
        self,
        input: Any,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """GEMINI API DOES NOT SUPPOR DIRECT COMPLETIONS

        Args:
            input (Any): Ignored
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Ignored
            kwargs (Any): Ignored

        Returns:
            None
        """

        warnings.warn("Gemini API does not support direct completions")
        return None

    def _format_output_structure(
        self,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
    ) -> Union[dict, None]:
        if isinstance(output_structure, dict):
            return output_structure

        if isinstance(output_structure, ModelMetaclass):
            # Output structure cannot be a ModelMetaclass for llama cpp server
            # Let's convert it to something OpenAI and llama cpp server both understand
            unrefed_schema = jsonref.replace_refs(output_structure.model_json_schema())  # type: ignore # noqa: E501

            def drop_dict_keys_recursive(input_dict: dict, drop_keys: list[str]) -> dict:
                output_dict = {}
                for key, value in input_dict.items():
                    if isinstance(value, dict):
                        output_dict[key] = drop_dict_keys_recursive(value, drop_keys)
                    elif key not in drop_keys:
                        output_dict[key] = value

                return output_dict

            properties = drop_dict_keys_recursive(
                unrefed_schema["properties"], ["title", "default"]
            )

            return {
                "type": "object",
                "properties": properties,
                "required": output_structure.model_json_schema()["required"],  # type: ignore # noqa: E501
            }

        return None

    def _create_chat_payload(
        self,
        chat: List[dict],
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        max_tokens: int = -1,
        **kwargs: Any,
    ) -> dict:
        """Create a payload for the Gemini API containing the chat and LLM options

        Args:
            chat (List[dict]): In form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Output structure to
                force. When using a pydantic model, only the reference should be passed.
                Correct = BaseLlamaCpp("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp("some text", MyPydanticModel())
                defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1 (=max).

        Returns:
            dict: payload to send to the Gemini API
        """

        # Format chat to Gemini API 'contents'
        contents = []
        for part in chat:
            contents.append({"parts": [{"text": part["content"]}], "role": part["role"]})

        # Make sure we have a structured output if requested
        if output_structure:
            response_schema = self._format_output_structure(output_structure)
            response_mime_type = "application/json"
        else:
            response_schema = None
            response_mime_type = "text/plain"

        # Make sure max_tokens is set correctly
        if max_tokens < 0:
            max_tokens = self.context_size

        # Set the generation config
        generation_config = kwargs
        generation_config["maxOutputTokens"] = max_tokens
        generation_config["responseMimeType"] = response_mime_type
        generation_config["responseSchema"] = response_schema

        return {
            "contents": contents,
            "generationConfig": generation_config,
        }

    def _stream_iterator(self, stream: requests.models.Response) -> Iterator[dict]:
        """Generator for streaming LLM response

        Args:
            stream (requests.models.Response): Reponse from the Gemini API

        Yields:
            Iterator[dict]: {"choices": [{"text": text}]}
        """

        chunk = None
        for line in stream.iter_lines():
            if (
                line.decode("utf-8") == '  "candidates": ['
                or line.decode("utf-8") == '  "error": {'
            ):
                if chunk is not None:
                    chunk_content = json.loads(chunk[:-2])
                    if chunk_content["candidates"][0].get("finishReason", "").lower() != "safety":
                        text = chunk_content["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        warnings.warn("Gemini did not respond due to 'safety reasons'")
                        text = ""
                    yield {"choices": [{"text": text}]}

                chunk = "{" + line.decode("utf-8")
            elif chunk is not None:
                chunk += line.decode("utf-8")

        if chunk is not None:
            chunk_content = json.loads(chunk[:-1])
            text = chunk_content["candidates"][0]["content"]["parts"][0]["text"]
            yield {"choices": [{"text": text}]}

    def _process_response(
        self,
        response: requests.models.Response,
        tries: int,
        stream: bool,
    ) -> str:
        """Processes the Gemini API response

        Args:
            response (requests.models.Response): Gemini API response
            tries (int): Number of API calls tried so far
            stream (bool): If True, streams the results to console. Defaults to False.

        Raises:
            Exception: If 'tries' 'exceeds self.max_tries'
            Exception: If Gemini API response code >=400 and <500

        Returns:
            str: Model response
        """

        if response.status_code == 408 or response.status_code == 503:
            if tries > self.max_retries:
                raise Exception(
                    f"Max retries exceeded. error code {response.status_code}: {response.json()['error']['message']}"  # noqa: E501
                )
            else:
                time.sleep(1 / self.rate_limit)
                results = ""

        elif response.status_code >= 400 and response.status_code < 600:
            raise Exception(
                f"error code {response.status_code}: {response.json()['error']['message']}"
            )

        elif response.status_code == 200:
            if stream:
                results = self.print_streamed(self._stream_iterator(response))  # type: ignore
            else:
                if response.json()["candidates"][0].get("finishReason", "").lower() != "safety":
                    results = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    warnings.warn("Gemini did not respond due to 'safety reasons'")
                    results = ""

        return results

    def run_chat(
        self,
        chat: List[dict],
        max_tokens: int = -1,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Runs a chat through the LLM

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1 (=max).
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Output structure to
                force. When using a pydantic model, only the reference should be passed.
                Correct = BaseLlamaCpp("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp("some text", MyPydanticModel())
                defaults to None.
            stream (bool, optional): If True, streams the results to console. Defaults to False.
            kwargs (Any): Any keyword arguments to add to the lmm call.
                See https://ai.google.dev/api/generate-content#generationconfig

        Returns:
            str: Generated content
        """

        cache_results = self.check_cache(chat, input_ischat=True)
        if cache_results:  # Check cache first
            results = cache_results
            if stream:
                print(results)
        else:  # Use LLM if not in cache
            # Make sure max_tokens is set correctly
            if max_tokens < 0:
                max_tokens = self.context_size

            # Different URL depending on streaming of not
            if stream:
                url = (
                    urljoin(self.url, self.model_name)  # type: ignore
                    + ":streamGenerateContent?key="
                    + self.gemini_api_key
                )
            else:
                url = (
                    urljoin(self.url, self.model_name)  # type: ignore
                    + ":generateContent?key="
                    + self.gemini_api_key
                )

            headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

            # Create the payload according to the Gemini API format
            payload = self._create_chat_payload(chat, output_structure, max_tokens, **kwargs)

            tries = 0
            while tries <= self.max_retries:
                # Make sure we don't exceed the request rate
                self._sleep_if_rate_limited()

                response = requests.post(url, headers=headers, json=payload, stream=stream)

                # Process the response, getting text back
                results = self._process_response(response, tries, stream)

                if response.status_code == 200:
                    break
                else:
                    tries += 1

            # And add the result to cache
            self.write_item_to_cache(chat, results, input_ischat=True)

        return results

    async def a_run_chat(
        self,
        chat: List[dict],
        max_tokens: int = -1,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Runs a chat through the LLM asynchronously

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Output structure to
                force. When using a pydantic model, only the reference should be passed.
                Correct = BaseLlamaCpp("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp("some text", MyPydanticModel())
                defaults to None.
            stream (bool, optional): If True, streams the results to console. Defaults to False.
            kwargs (Any): Any keyword arguments to add to the lmm call.
                See https://ai.google.dev/api/generate-content#generationconfig

        Returns:
            str: Generated content
        """

        cache_results = self.check_cache(chat, input_ischat=True)
        if cache_results:  # Check cache first
            results = cache_results
        else:  # Use LLM if not in cache
            # Make sure max_tokens is set correctly
            if max_tokens < 0:
                max_tokens = self.context_size

            url = urljoin(self.url, self.model_name) + ":generateContent?key=" + self.gemini_api_key  # type: ignore  # noqa: E501
            headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

            # Create the payload according to the Gemini API format
            payload = self._create_chat_payload(chat, output_structure, max_tokens, **kwargs)

            tries = 0
            while tries <= self.max_retries:
                # Make sure we don't exceed the rate limit
                time.sleep(1 / self.rate_limit)

                response = await asyncio.to_thread(
                    requests.post, url, headers=headers, json=payload, stream=stream
                )

                # Process results, getting text back
                results = self._process_response(response, tries, stream)

                if response.status_code == 200:
                    break
                else:
                    tries += 1

            # And add the result to cache
            self.write_item_to_cache(chat, results, input_ischat=True)

        return results

    def num_chat_tokens(self, chat: List[dict]) -> int:
        """Return the length of the tokenized chat using the Gemini API

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

        Raises:
            Exception: If Gemini API response code >=400 and <500

        Returns:
            int: number of tokens
        """

        url = urljoin(self.url, self.model_name) + ":countTokens?key=" + self.gemini_api_key  # type: ignore  # noqa: E501
        payload = {"contents": self._create_chat_payload(chat)["contents"]}
        headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            num_tokens = response.json()["totalTokens"]
        elif response.status_code >= 400 and response.status_code < 500:
            # Error
            raise Exception(
                f"error code {response.status_code}: {response.json()['error']['message']}"
            )

        return num_tokens

    def tokenize(self, content: str) -> Union[List[str], List[int]]:
        """Tokenize a string. Approximates using tiktoken but should be good enough.

        Args:
            content (str): String to tokenize

        Returns:
            Union[List[str], List[int]]: Tokenized string
        """

        return self.tokenizer.tokenize(content)

    def untokenize(self, tokens: Union[List[str], List[int]]) -> str:
        """Generate a string from a list of tokens.  Approximates using tiktoken but
        should be good enough.

        Args:
            tokens (Union[List[str], List[int]]): Tokenized string

        Returns:
            str: Untokenized string
        """

        if not tokens:
            return ""

        return self.tokenizer.convert_tokens_to_string(tokens)
