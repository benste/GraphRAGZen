import os
import sys
from typing import Any, List, Optional, Union

from graphragzen.llm.base_llm import LLM
from llama_cpp import Llama  # , LlamaCache
from pydantic import BaseModel
from pydantic._internal._fields import PydanticMetadata
from transformers import AutoTokenizer

from .typing import ChatNames, LlmLoadingConfig

# llama_prompter imports pydantic._internal._fields.PydanticGeneralMetadata
# but it should import pydantic._internal._fields.PydanticMetadata
# This hack fixes that by creating an alias
sys.modules["pydantic._internal._fields"].PydanticGeneralMetadata = PydanticMetadata
from llama_prompter import Prompter


# llama_prompter is used to create the grammar that forces a specific structure to the LLM output.
# It calls llama_cpp.llama_grammar.LlamaGrammar.from_string with verbosity to False, but sadly
# that function did not implement a verbosity check and still prints to the terminal.
# The following function will suppressing sys.stdout
def suppress_prompter_output(output_structure: BaseModel) -> Prompter:
    # Save the current stdout
    original_stdout = sys.stdout
    try:
        # Redirect stdout to null (suppress output)
        sys.stdout = open(os.devnull, "w")
        # Call the function
        result = Prompter("""{output:output_structure}""")
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

    return result


class BaseLlamCpp(LLM):
    """Loads a GGUF model using llama cpp python and it's corresponding tokenizer from HF"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, input: Any, output_structure: Optional[BaseModel] = None, **kwargs: Any
    ) -> Any:
        """Call the LLM as you would llm(input), but allow to force an output structure.

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

        if output_structure is not None:
            grammar = suppress_prompter_output(output_structure)._grammar
            kwargs.update({"grammar": grammar})

        return self.model(
            input,
            **kwargs,
        )

    def run_chat(
        self,
        chat: List[dict],
        max_tokens: int = -1,
        output_structure: Optional[BaseModel] = None,
        stream: bool = False,
    ) -> str:
        """Runs a chat through the LLM

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
            output_structure (BaseModel, optional): Output structure to force, e.g. grammars from
                llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the reference.
                Correct = BaseLlamCpp.run_chat("some text", MyPydanticModel)
                Wrong = BaseLlamCpp.run_chat("some text", MyPydanticModel())
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
            results = self(
                llm_input,
                output_structure=output_structure,
                stop=["<eos>"],
                echo=False,
                repeat_penalty=1.0,
                max_tokens=max_tokens,
                stream=stream,
            )

            if stream:
                results = self.print_streamed(results)  # type: ignore
            else:
                results = results["choices"][0]["text"]  # type: ignore

            # And add the result to cache
            self.write_item_to_cache(llm_input, results)

        return results

    def num_chat_tokens(self, chat: List[dict]) -> int:
        """Return the length of the tokenized chat

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...

        Returns:
            int: number of tokens
        """

        return len(
            self.tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
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


class Gemma2GGUF(BaseLlamCpp):
    """Loads the GGUF version of a gemma2 model using llama-cpp-python"""

    def __init__(self, **kwargs: Union[dict, LlmLoadingConfig, Any]) -> None:
        config = LlmLoadingConfig(**kwargs)  # type: ignore
        self.model = Llama(
            model_path=config.model_storage_path, verbose=False, n_ctx=config.context_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_URI)
        self.chatnames = ChatNames(user="user", model="assistant")
        self.config = config

        super().__init__()


class Phi35MiniGGUF(BaseLlamCpp):
    """Loads the GGUF version of a Phi 3.5 Mini model using llama-cpp-python"""

    def __init__(self, **kwargs: Union[dict, LlmLoadingConfig, Any]):
        config = LlmLoadingConfig(**kwargs)  # type: ignore
        self.model = Llama(
            model_path=config.model_storage_path, verbose=False, n_ctx=config.context_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_URI)
        self.chatnames = ChatNames(system="system", user="user", model="assistant")
        self.config = config

        super().__init__()
