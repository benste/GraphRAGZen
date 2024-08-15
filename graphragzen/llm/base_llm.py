from copy import deepcopy
from datetime import datetime
from typing import Any, Iterator, List

from .typing import ChatNames


class LLM:
    model: Any = None
    tokenizer: Any = None
    chatnames: ChatNames = ChatNames()

    def run_chat(self, chat: List[dict], max_tokens: int = -1, stream: bool = False) -> str:
        """Runs a chat through the LLM

        Args:
            chat (List[dict]): in form [{"role": ..., "content": ...}, {"role": ..., "content": ...
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to -1.
            stream (bool, optional): If True, streams the results to console. Defaults to False.

        Returns:
            str: Generated content
        """
        return ""

    def tokenize(self, content: str) -> List[str]:
        """Tokenize a string

        Args:
            content (str): String to tokenize

        Returns:
            List[str]: Tokenized string
        """
        return [""]

    def untokenize(self, tokens: List[str]) -> str:
        """Generate a string from a list of tokens

        Args:
            tokens (List[str]): Tokenized string

        Returns:
            str: Untokenized string
        """
        return ""

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
            token = s["choices"][0]["text"]
            print(token, end="", flush=True)
            full_text += token
            num_tokens += 1

        elapsed_time = datetime.now() - start
        if timeit:
            print(f"tokens / sec = {num_tokens / elapsed_time.seconds}")

        return full_text

    def format_chat(self, chat: List[tuple], established_chat: List[dict] = []) -> List[dict]:
        """format chat with the correct names ready for `tokenizer.apply_chat_template`

        Args:
            chat (List[tuple]): [(role: content), (role: content)]
                                role (str): either "user" or "model"
                                content (str)
            established_chat (List[dict], optional): Already established chat to append to.
                Defaults to [].

        Returns:
            List[dict]: [{"role": ..., "content": ...}, {"role": ..., "content": ...}]
        """

        # Make sure we don't change the input variable
        formatted_chat = deepcopy(established_chat)

        if len(formatted_chat) == 0 and chat[0][0] != "user":
            # First role MUST be user
            # TODO: only show warnings if requested
            # warnings.warn("Chat did not start with 'user', adding `'user': ' '` to the chat")
            chat = [("user", " ")] + chat

        for role, content in chat:
            formatted_chat.append({"role": self.chatnames.model_dump()[role], "content": content})

        return formatted_chat
