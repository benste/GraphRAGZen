import asyncio
import json
import os
from typing import Any, List, Optional, Union
from urllib.parse import urljoin

import jsonref
import requests
import tiktoken
from graphragzen.llm.base_llm import LLM
from openai import AsyncOpenAI, OpenAI
from pydantic._internal._model_construction import ModelMetaclass
from transformers import AutoTokenizer


class OpenAICompatibleClient(LLM):
    """Uses API enpoints that are compatible with the OpenAI API enpoints"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = "placeholder_model_name",
        context_size: int = 8192,
        api_key_env_variable: str = "OPENAI_API_KEY",
        openai_organization_id: Optional[str] = None,
        openai_project_id: Optional[str] = None,
        hf_tokenizer_URI: Optional[str] = None,
        max_retries: int = 3,
        use_cache: bool = True,
        cache_persistent: bool = True,
        persistent_cache_file: str = "./phi35_mini_persistent_cache.yaml",
    ) -> None:
        """
        Args:
            base_url (str, optional): url with API endpoints. Not needed if using openAI.
                Defaults to None.
            model_name (str, optional): Name of the model to use. Required when using openAI API.
                Defaults to "placeholder_model_name".
            context_size (int): Context size of the model. Defaults to 8192.
            api_key_env_variable (str): Environment variable to read the openAI API key from.
                Defaults to "OPENAI_API_KEY".
            openai_organization_id (str, optional): Organization ID to use when querying the openAI
                API. Defaults to None.
            openai_project_id (str, optional): Project ID to use when querying the openAI API.
                Defaults to None.
            hf_tokenizer_URI (str, optional): The URI to a tokenizer on HuggingFace. If not provided
                the API will be tested on the ability to tokenize. If that also fails a tiktoken is
                initiated.
            max_retries (optional, int): Number of times to retry on timeout. Defaults to 3.
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """

        self.context_size = context_size
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_persistent = cache_persistent
        self.persistent_cache_file = persistent_cache_file
        self._initiate_tokenizer(hf_tokenizer_URI, base_url, model_name)

        openai_api_key = os.environ.get(api_key_env_variable)
        if not openai_api_key:
            # Set a fake key that will load the client but won't authenticate to openAI
            openai_api_key = "sk-proj-ZNBiANHU9ilDCsySC0fVeTD76Vc5DNPbstPThgSdtgI0QV51FBBLeNgL2IZVEFfNX548MqAhnAWfJnlEQtzCelsGL73fx8NBuxc3ZTrP8Ux6qgX4c4emqcIrGTnz"  # noqa: E501

        self.client: OpenAI = OpenAI(
            base_url=base_url,
            api_key=openai_api_key,
            organization=openai_organization_id,
            project=openai_project_id,
            max_retries=max_retries,
        )

        self.async_client: AsyncOpenAI = AsyncOpenAI(
            base_url=base_url,
            api_key=openai_api_key,
            organization=openai_organization_id,
            project=openai_project_id,
            max_retries=max_retries,
        )

        super().__init__()

    def __call__(
        self,
        input: Any,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM as you would llm(input) for simple completion.

        Args:
            input (Any): Any input you would normally pass to llm(input, kwargs)
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Ignored
            kwargs (Any): Any keyword arguments you would normally pass to llm(input, kwargs)

        Returns:
            Any
        """

        if asyncio.get_event_loop().is_running():
            client = self.async_client
        else:
            client = self.client  # type: ignore

        result = client.completions.create(
            prompt=input,
            model=self.model_name,
            **kwargs,
        )

        # Format and return
        return {"choices": [{"text": result.content}]}  # type: ignore

    def _initiate_tokenizer(
        self,
        hf_tokenizer_URI: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Tries to use the following methods of tokenization in order:
        1. Load tokenizer from HF using hf_tokenizer_URI
        2. Try to tokenize and de-tokenize using the API endpoints base_url/tokenize
            and base_url/detokenize
        3. Try to initiate tiktoken, getting encoding from
            tiktoken.encoding_for_model(model_name)

        If any tokenizer is success initiated it is set as the class tokenizer and the other methods
        are not tried.
        """

        tokenizer_set = False

        # Try to load tokenizer from HF
        if hf_tokenizer_URI:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_URI)
                tokens = self.tokenizer.tokenize("hello world, how are you?")
                self.tokenizer.convert_tokens_to_string(tokens)
                tokenizer_set = True
            except Exception:
                tokenizer_set = False

        # Try API tokenizer and detokenizer endpoints
        if not tokenizer_set and base_url:
            try:
                self.tokenizer = ApiTokenizer(base_url)
                tokens = self.tokenizer.tokenize("hello world, how are you?")
                self.tokenizer.convert_tokens_to_string(tokens)
                tokenizer_set = True
            except Exception:
                tokenizer_set = False

        # Try to load and use tiktoken tokenizer
        if not tokenizer_set and model_name:
            try:
                self.tokenizer = TikTokenTokenizer(model_name)
                tokens = self.tokenizer.tokenize("hello world, how are you?")
                self.tokenizer.convert_tokens_to_string(tokens)
                tokenizer_set = True
            except Exception:
                tokenizer_set = False

        if not tokenizer_set:
            raise Exception(
                "Failed to load any of huggingface tokenizer, API tokenizer and tiktoken"
            )

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
            properties = json.loads(json.dumps(unrefed_schema["properties"], indent=2))
            return {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": output_structure.model_json_schema()["required"],  # type: ignore # noqa: E501
                },
            }

        return None

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
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
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

        cache_results = self.check_cache(str(chat))
        if cache_results:  # Check cache first
            results = cache_results
        else:  # Use LLM if not in cache
            # Make sure max_tokens is set correctly
            if max_tokens < 0:
                max_tokens = 10**10

            # Though OpenAI API can handle Pydantic ModelMetaClass, not all API's can. Let's
            # transform it into a dict most OpenAI compatible API's can handle
            response_format = self._format_output_structure(output_structure)

            results = self.client.chat.completions.create(  # type: ignore
                messages=chat,
                model=self.model_name,
                response_format=response_format,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )

            if stream:
                results = self.print_streamed(results)  # type: ignore
            else:
                results = results.choices[0].message.content  # type: ignore

            # And add the result to cache
            self.write_item_to_cache(str(chat), results)

        return results

    async def a_run_chat(
        self,
        chat: List[dict],
        max_tokens: int = -1,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Runs a chat through the LLM asynchonously

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
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

        cache_results = self.check_cache(str(chat))
        if cache_results:  # Check cache first
            content = cache_results
        else:  # Use LLM if not in cache
            # Make sure max_tokens is set correctly
            if max_tokens < 0:
                max_tokens = 10**10

            # Though OpenAI API can handle Pydantic ModelMetaClass, not all API's can. Let's
            # transform it into a dict most OpenAI compatible API's can handle
            response_format = self._format_output_structure(output_structure)

            results = await self.async_client.chat.completions.create(  # type: ignore
                messages=chat,
                model=self.model_name,
                response_format=response_format,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )

            content = results.choices[0].message.content  # type: ignore

            # And add the result to cache
            self.write_item_to_cache(str(chat), content)

        return content

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
            return len(
                self.tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            # Otherwise we'll join the messages and which will lead to a relatively good estimate
            # of the final prompt string
            formatted_chat = " ".join(
                [f"{message['role']} {message['content']}" for message in chat]
            )
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


class OllamaClient(OpenAICompatibleClient):
    """Interact with an LLM running on an Ollama Server.


    !!During inference this class can force json output, but not the structure of the json.!!

    The only reason this class exists is because the Ollama server does not have the
    'response_format' feature; it can only be forced to output some json, but not force its
    structure. Forcing an output structure significantly increases the quality of generated
    graphs, concider using llama.cpp for serving your model.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        """Interact with an LLM running on an Ollama Server.

        !!During inference this class can force json output, but not the structure of the json.!!

        The only reason this class exists is because the Ollama server does not have the
        'response_format' feature; it can only be forced to output some json, but not force its
        structure. Forcing an output structure significantly increases the quality of generated
        graphs, concider using llama.cpp for serving your model.

        Args:
            base_url (str, optional): url with API endpoints. Not needed if using openAI.
                Defaults to None.
            model_name (str, optional): Name of the model to use. Required when using openAI API.
                Defaults to "placeholder_model_name".
            context_size (int): Context size of the model. Defaults to 8192.
            api_key_env_variable (str): Environment variable to read the openAI API key from.
                Defaults to "OPENAI_API_KEY".
            openai_organization_id (str, optional): Organization ID to use when querying the openAI
                API. Defaults to None.
            openai_project_id (str, optional): Project ID to use when querying the openAI API.
                Defaults to None.
            hf_tokenizer_URI (str, optional): The URI to a tokenizer on HuggingFace. If not provided
                the API will be tested on the ability to tokenize. If that also fails a tiktoken is
                initiated.
            max_retries (optional, int): Number of times to retry on timeout. Defaults to 2.
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """
        super().__init__(*args, **kwargs)

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
            max_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to -1 (infinite).
            output_structure (ModelMetaclass, optional): Output structure to force. Ollama
                can only force some json, not the structure of the json. Making this non-empty
                forces a json output. Defaults to None.
            stream (bool, optional): If True, streams the results to console. Defaults to False.
            kwargs (Any): Any keyword arguments to add to the lmm call.

        Returns:
            str: Generated content
        """
        if output_structure:
            output_structure = {"type": "json_object"}

        return super().run_chat(chat, max_tokens, output_structure, stream, **kwargs)

    async def a_run_chat(
        self,
        chat: List[dict],
        max_tokens: int = -1,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Runs a chat through the LLM asynchonously

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate.
                Defaults to -1 (infinite).
            output_structure (ModelMetaclass, optional): Output structure to force. Ollama
                can only force some json, not the structure of the json. Making this non-empty
                forces a json output. Defaults to None.
            stream (bool, optional): Placeholder for compatibility with sync version, not used.
            kwargs (Any): Any keyword arguments to add to the lmm call.

        Returns:
            str: Generated content
        """

        if output_structure:
            output_structure = {"type": "json_object"}

        return await super().a_run_chat(chat, max_tokens, output_structure, stream, **kwargs)


class ApiTokenizer:
    """tokenizes and detokenizes using API endpoints"""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def tokenize(self, content: str, base_url: Optional[str] = None) -> List[int]:
        base_url = base_url or self.base_url
        result = requests.post(urljoin(base_url, "tokenize"), json={"content": content})
        return result.json().get("tokens")

    def convert_tokens_to_string(self, tokens: List[int], base_url: Optional[str] = None) -> str:
        base_url = base_url or self.base_url
        result = requests.post(urljoin(base_url, "detokenize"), json={"tokens": tokens})
        return result.json().get("content")


class TikTokenTokenizer:
    """tokenizes and detokenizes using tiktoken library"""

    def __init__(self, model_name: str):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def tokenize(self, content: str) -> List[int]:
        return self.encoding.encode(content)

    def convert_tokens_to_string(self, tokens: List[int]) -> str:
        return self.encoding.decode(tokens)
