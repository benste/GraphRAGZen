from typing import List

from pydantic import BaseModel

from graphragzen.prompts.default_prompts import entity_extraction_prompts
  
  
class EntityExtractionPromptFormatting(BaseModel):
    tuple_delimiter: str = "<|>"  # delimiter between tuples in an output record, default is '<|>'
    record_delimiter: str = "##"  # delimiter between records, default is '##'
    completion_delimiter: str = "<|COMPLETE|>"
    entity_types: List[str] = ["organization", "person", "geo", "event"]
    input_text: str = None
    
class EntityExtractionPrompts(BaseModel):
    entity_extraction_prompt: str = entity_extraction_prompts.ENTITY_EXTRACTION_PROMPT
    continue_prompt: str = entity_extraction_prompts.CONTINUE_PROMPT
    loop_prompt: str = entity_extraction_prompts.LOOP_PROMPT
    
class EntityExtractionPromptConfig(BaseModel):
    prompts: EntityExtractionPrompts
    formatting: EntityExtractionPromptFormatting
    
class EntityExtractionConfig(BaseModel):
    max_gleans: int = 5
    column_to_extract: str = 'chunk'
    results_column: str = 'raw_entities'
    
class RawEntitiesToGraphConfig(BaseModel):
    raw_entities_column: str = 'raw_entities'
    reference_column: str = 'chunk_id'  # source_id will be added to the edged and nodes. This allows source reference when quiring the graph
    feature_delimiter: str = "\n"  # When the same node or edge is found multiple times, features are concatenated using this demiliter
