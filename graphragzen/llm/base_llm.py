import os
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Iterator, List, Optional, Union

import yaml
from pydantic._internal._model_construction import ModelMetaclass

from .typing import ChatNames


@dataclass
class LLM(ABC):
    """Base class to communicate with local or remote LLM's

    Be carefull when using the same persistent cache file while switching or updating models or
    tokenizers. The LLM will search for cached llm-in -> llm-out in in the cache file and not
    re-process input.
    """

    context_size = 0
    use_cache = True
    cache_persistent = False
    persistent_cache_file = ""
    model_name: Any = None
    tokenizer: Any = None
    chatnames: ChatNames = ChatNames()

    def __init__(self) -> None:
        self._initiate_cache()

    @abstractmethod
    def __call__(
        self,
        input: Any,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM as you would llm(input)

        Args:
            input (Any): Any input you would normally pass to llm(input, kwargs)
            output_structure (Optional[Union[ModelMetaclass, dict]], optional): Output structure to
                force. e.g. grammars from llama.cpp. When using a pydantic model, only the reference
                should be passed.
                Correct = BaseLlamaCpp("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp("some text", MyPydanticModel())
            kwargs (Any): Any keyword arguments you would normally pass to llm(input, kwargs)

        Returns:
            Any
        """
        pass

    @abstractmethod
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
        return ""

    @abstractmethod
    def tokenize(self, content: str) -> Union[List[str], List[int]]:
        """Tokenize a string

        Args:
            content (str): String to tokenize

        Returns:
            List[str]: Tokenized string
        """
        return [""]

    @abstractmethod
    def untokenize(self, tokens: List[str]) -> str:
        """Generate a string from a list of tokens

        Args:
            tokens (List[str]): Tokenized string

        Returns:
            str: Untokenized string
        """
        return ""

    @abstractmethod
    def num_chat_tokens(self, chat: List[dict]) -> int:
        """Return the length of the tokenized chat

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

        Returns:
            int: number of tokens
        """
        return 0

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
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to None
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

        return self.run_chat(chat, max_tokens, output_structure, stream, **kwargs)

    def print_streamed(self, stream: Iterator, timeit: bool = False) -> str:
        """Streams the generated tokens to the terminal and returns the full generated text.

        Args:
            stream (Iterator)
            timeit (bool, optional): If True display the number of tokens generated / sec.
                Defaults to False.

        Returns:
            str: Generated text
        """
        full_text = ""
        start = datetime.now()
        num_tokens = 0
        for s in stream:
            try:
                # llama-cpp-python output
                token = s["choices"][0]["text"]
            except TypeError:
                # OpenAI compatible output
                token = s.choices[0].delta.content

            print(token, end="", flush=True)
            if token:
                full_text += token
            num_tokens += 1

        elapsed_time = datetime.now() - start
        if timeit:
            print(f"tokens / sec = {num_tokens / elapsed_time.seconds}")

        return full_text

    def format_chat(self, chat: List[tuple], established_chat: List[dict] = []) -> List[dict]:
        """format chat with the correct names ready for `tokenizer.apply_chat_template`

        Args:
            chat (List[tuple]): [(role, content), (role, content)]
                                - role (str): either "system", "user" or "model"
                                - content (str)
            established_chat (List[dict], optional): Already formatted chat to append to.
                Defaults to [].

        Returns:
            List[dict]: [{"role": ..., "content": ...}, {"role": ..., "content": ...}]
        """

        # Make sure we don't change the input variable
        formatted_chat = deepcopy(established_chat)

        # First role MUST be user or system
        if len(formatted_chat) == 0 and chat[0][0] not in ["user", "system"]:
            # TODO: only show warnings if requested
            # warnings.warn("Chat did not start with 'user', adding `'user': ' '` to the chat")
            chat = [("user", " ")] + chat

        for role, content in chat:
            formatted_chat.append({"role": self.chatnames.model_dump()[role], "content": content})

        return formatted_chat

    def check_cache(self, llm_input: str) -> Union[str, None]:
        """Checks the hash(llm_in) -> llm_out cache and returns stored output if found.

        Args:
            llm_input (str): To check in cache for existing cached output.

        Returns:
            Union[str, None]
        """
        if self.cache:
            return self.cache.get(sha256(llm_input.encode("utf-8")).hexdigest(), None)

        return None

    def write_item_to_cache(self, llm_input: str, llm_output: str) -> None:
        """If a persistent cache file exists, this function can be used to append llm output to it.

        Args:
            llm_input (str)
            llm_output (str)
        """
        if self.use_cache:
            hash = sha256(llm_input.encode("utf-8")).hexdigest()
            self.cache.update({hash: llm_output})

            if self.cache_persistent:
                with open(self.persistent_cache_file, "a") as cache_file:
                    cache_file.write("\n")
                    cache_file.write(yaml.dump({hash: llm_output}))

    def _initiate_cache(self) -> None:
        """If requested creates an in-memory hash(llm_in) -> llm_out cache. Additionally, if
        requested, reads and writes this cache persistently to disk.
        """
        self.cache = None

        if self.use_cache:
            self.cache = {}

            if self.cache_persistent and self.persistent_cache_file:
                # Load or create persisten cache file
                if not os.path.isfile(self.persistent_cache_file):
                    os.makedirs(os.path.dirname(self.persistent_cache_file) or "./", exist_ok=True)

                else:
                    with open(self.persistent_cache_file, "r") as stream:
                        self.cache = yaml.safe_load(stream)

                    if not self.cache:
                        # Might have loaded an empty file
                        self.cache = {}

            else:
                warnings.warn(
                    f"""LLM initiated with persisten cache but invalid persistent cache file
                    provided.
                    Persistent cache file provided: {self.persistent_cache_file}"""
                )
                self.cache_persistent = False
