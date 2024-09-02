from ..typing.MappedBaseModel import MappedBaseModel


class ChatNames(MappedBaseModel):
    """The ChatNames the LLM expects in the prompt

    Args:
        user (str, optional): Name of the user. Defaults to 'user'.
        model (str, optional): Name of the model Defaults to 'assistant'.
    """

    system: str = "system"
    user: str = "user"
    model: str = "model"
