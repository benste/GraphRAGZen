import os
import sys
from typing import Any, List, Optional, Union

from graphragzen.llm.base_llm import LLM
from llama_cpp import Llama, LlamaGrammar
from pydantic._internal._fields import PydanticMetadata
from pydantic._internal._model_construction import ModelMetaclass
from transformers import AutoTokenizer

from .typing import ChatNames

# llama_prompter imports pydantic._internal._fields.PydanticGeneralMetadata
# but it should import pydantic._internal._fields.PydanticMetadata
# This hack fixes that by creating an alias
sys.modules["pydantic._internal._fields"].PydanticGeneralMetadata = PydanticMetadata  # type: ignore
from llama_prompter import Prompter  # noqa: F401, E402


# llama_prompter is used to create the grammar that forces a specific structure to the LLM output.
# It calls llama_cpp.llama_grammar.LlamaGrammar.from_string with verbosity to False, but sadly
# that function did not implement a verbosity check and still prints to the terminal.
# The following function will suppressing sys.stdout
def suppress_prompter_output(
    output_structure: Union[ModelMetaclass, dict]
) -> Union[LlamaGrammar, dict, None]:
    if isinstance(output_structure, dict):
        return output_structure

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

    return result._grammar


class BaseLlamaCpp(LLM):
    """Loads a GGUF model using llama cpp python and it's corresponding tokenizer from HF"""

    def __init__(
        self,
        model_storage_path: str,
        tokenizer_URI: str,
        context_size: int = 8192,
        n_gpu_layers: int = -1,
        use_cache: bool = True,
        cache_persistent: bool = True,
        persistent_cache_file: str = "./llm_persistent_cache.yaml",
    ) -> None:
        """Initiate a llama cpp model

        Args:
            model_storage_path (str): Path to the model on the local filesystem
            tokenizer_URI (str): URI for the tokenizer
            context_size (int, optional): Size of the context window in tokens. Defaults to 8192
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            n_gpu_layers (int, optional): Number of layers to offload to GPU (-ngl). If -1, all
                layers are offloaded. You need to install llama-cpp-python with the correct cuda
                support. Out of the box GraphRAGZen's llama-cpp-python is the CPU version only.
                Defaults to -1.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """

        self.context_size = context_size
        self.use_cache = use_cache
        self.cache_persistent = cache_persistent
        self.persistent_cache_file = persistent_cache_file

        self.model = Llama(
            model_path=model_storage_path,
            verbose=False,
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_URI)

        if not self.chatnames:
            self.chatnames = ChatNames()

        super().__init__()

    def __call__(
        self,
        input: Any,
        output_structure: Optional[Union[ModelMetaclass, dict]] = None,
        **kwargs: Any,
    ) -> Any:
        """Call the LLM as you would llm(input), but allow to force an output structure.

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

        if output_structure is not None:
            grammar = suppress_prompter_output(output_structure)
            kwargs.update({"grammar": grammar})

        return self.model(
            input,
            **kwargs,
        )

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
            output_structure (ModelMetaclass, optional): Output structure to force, e.g. grammar
                from llama.cpp. This SHOULD NOT be an instance of the pydantic model, just the
                reference.
                Correct = BaseLlamaCpp.run_chat("some text", MyPydanticModel)
                Wrong = BaseLlamaCpp.run_chat("some text", MyPydanticModel())
            stream (bool, optional): If True, streams the results to console. Defaults to False.
            kwargs (Any): Any keyword arguments to add to the lmm call.

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
                input=llm_input,
                output_structure=output_structure,
                stop=["<eos>"],
                echo=False,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
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

    def tokenize(self, content: str) -> Union[List[str], List[int]]:
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


class Gemma2GGUF(BaseLlamaCpp):
    """Loads the GGUF version of a gemma2 model using llama-cpp-python"""

    def __init__(
        self,
        model_storage_path: str,
        tokenizer_URI: str,
        context_size: int = 8192,
        n_gpu_layers: int = -1,
        use_cache: bool = True,
        cache_persistent: bool = True,
        persistent_cache_file: str = "./llm_persistent_cache.yaml",
    ) -> None:
        """Load the GGUF version of a gemma2 model using llama-cpp-python and it's corresponding
        tokenizer.

        Args:
            model_storage_path (str): Path to the model on the local filesystem
            tokenizer_URI (str): HuggingFace URI for the tokenizer
            context_size (int, optional): Size of the context window in tokens. Defaults to 8192
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            n_gpu_layers (int, optional): Number of layers to offload to GPU (-ngl). If -1, all
                layers are offloaded. You need to install llama-cpp-python with the correct cuda
                support. Out of the box GraphRAGZen's llama-cpp-python is the CPU version only.
                Defaults to -1.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """

        self.chatnames = ChatNames(user="user", model="assistant")

        super().__init__(
            model_storage_path,
            tokenizer_URI,
            context_size,
            n_gpu_layers,
            use_cache,
            cache_persistent,
            persistent_cache_file,
        )


class Phi35MiniGGUF(BaseLlamaCpp):
    """Loads the GGUF version of a Phi 3.5 Mini model using llama-cpp-python"""

    def __init__(
        self,
        model_storage_path: str,
        tokenizer_URI: str,
        context_size: int = 8192,
        n_gpu_layers: int = -1,
        use_cache: bool = True,
        cache_persistent: bool = True,
        persistent_cache_file: str = "./llm_persistent_cache.yaml",
    ) -> None:
        """Load the GGUF version of a Phi 3.5 Mini model using llama-cpp-python and it's
        corresponding tokenizer.

        Args:
            model_storage_path (str): Path to the model on the local filesystem
            tokenizer_URI (str): HuggingFace URI for the tokenizer
            context_size (int, optional): Size of the context window in tokens. Defaults to 8192
            use_cache (bool, optional): Use a cache to find output for previously processed inputs
                in stead of re-generating output from the input. Default to True.
            n_gpu_layers (int, optional): Number of layers to offload to GPU (-ngl). If -1, all
                layers are offloaded. You need to install llama-cpp-python with the correct cuda
                support. Out of the box GraphRAGZen's llama-cpp-python is the CPU version only.
                Defaults to -1.
            cache_persistent (bool, optional): Append the cache to a file on disk so it can be
                re-used between runs. If False will use only in-memory cache. Default to True
            persistent_cache_file (str, optional): The file to store the persistent cache.
                Defaults to './llm_persistent_cache.yaml'.
        """

        self.chatnames = ChatNames(system="system", user="user", model="assistant")

        super().__init__(
            model_storage_path,
            tokenizer_URI,
            context_size,
            n_gpu_layers,
            use_cache,
            cache_persistent,
            persistent_cache_file,
        )
