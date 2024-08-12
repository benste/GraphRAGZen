from typing import List, Optional
from pydantic import BaseModel

from graphragzen.prompts.default_prompts import summarization_prompts
  
  
class DescriptionSummarizationPromptFormatting(BaseModel):
    entity_name: Optional[str]
    description_list: Optional[List]
    
class DescriptionSummarizationPromptConfig(BaseModel):
    prompt: str = summarization_prompts.SUMMARIZE_PROMPT
    formatting: DescriptionSummarizationPromptFormatting


class DescriptionSummarizationConfig(BaseModel):
    feature_delimiter: str = "\n"  # When the same node or edge is found multiple times, features are concatenated using this demiliter
    max_input_tokens: int = 4000
    max_output_tokens: int = 500