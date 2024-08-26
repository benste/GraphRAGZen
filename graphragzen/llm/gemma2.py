from typing import Any, List, Union

from graphragzen.llm.base_llm import LLM
from llama_cpp import Llama  # , LlamaCache
from transformers import AutoTokenizer

from .typing import ChatNames, LlmLoadingConfig


class Gemma2GGUF(LLM):
    """Loads the GGUF version of a gemma2 model using llama-cpp-python"""

    def __init__(self, **kwargs: Union[dict, LlmLoadingConfig, Any]):
        config = LlmLoadingConfig(**kwargs)  # type: ignore
        self.model = Llama(
            model_path=config.model_storage_path, verbose=False, n_ctx=config.context_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_URI)
        self.chatnames = ChatNames(user="user", model="assistant")
        self.config = config

        super().__init__()

    def run_chat(self, chat: List[dict], max_tokens: int = -1, stream: bool = False) -> str:
        """Runs a chat through the LLM

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
            stream (bool, optional): If True, streams the results to console. Defaults to False.

        Returns:
            str: Generated content
        """

        llm_input = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        llm_input = llm_input.removeprefix("<bos>")

        # Check cache first
        cache_results = self.check_cache(llm_input)
        if cache_results:
            results = cache_results
        else:
            # Use LLM if not in cache
            results = self.model(
                llm_input,
                stop=["<eos>"],
                echo=False,
                repeat_penalty=1.0,
                max_tokens=max_tokens,
                stream=stream,
            )
            # And add the result to cache
            self.write_item_to_cache(llm_input, results)

            if stream:
                results = self.print_streamed(results)  # type: ignore
            else:
                results = results["choices"][0]["text"]  # type: ignore

        return results

    def num_chat_tokens(self, chat: List[dict]) -> int:
        """Return the length of the tokenized chat

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

        Returns:
            int: number of tokens
        """

        return len(
            self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    def tokenize(self, content: str) -> List[str]:
        """Tokenize a string

        Args:
            content (str): String to tokenize

        Returns:
            List[str]: Tokenized string
        """

        return self.tokenizer.tokenize(content)

    def untokenize(self, tokens: List[str]) -> str:
        """Generate a string from a list of tokens

        Args:
            tokens (List[str]): Tokenized string

        Returns:
            str: Untokenized string
        """

        return self.tokenizer.convert_tokens_to_string(tokens)
