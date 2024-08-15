from typing import List

from graphragzen.llm.base_llm import LLM
from llama_cpp import Llama, LlamaCache
from transformers import AutoTokenizer

from .typing import ChatNames


class Gemma2GGUF(LLM):
    """Loads the GGUF version of a gemma2 model using llama-cpp-python"""

    def __init__(self, model_path: str, tokenizer_URI: str, context_size: int = 8192):
        self.model = Llama(model_path=model_path, verbose=False, n_ctx=context_size)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_URI)
        self.chatnames = ChatNames(user="user", model="assistant")
        self.model.set_cache(LlamaCache())
        self.cache = self.model.cache

    def run_chat(self, chat: List[dict], max_tokens: int = -1, stream: bool = False) -> str:
        # Somehow in a kedro pipeline the cache is not attached to the model anymore, et's re-attach
        self.model.set_cache(self.cache)

        llm_input = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        llm_input = llm_input.removeprefix("<bos>")
        results = self.model(
            llm_input,
            stop=["<eos>"],
            echo=False,
            repeat_penalty=1.0,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            results = self.print_streamed(results)
        else:
            results = results["choices"][0]["text"]

        return results

    def tokenize(self, content: str) -> List[str]:
        return self.tokenizer.tokenize(content)

    def untokenize(self, tokens: List[str]) -> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
