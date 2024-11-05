import json

from graphragzen.llm.base_llm import LLM

class MockLLM(LLM):
    def format_chat(self, *args, **kwargs):
        return [{"user": "This is a chat"}]
    
    def run_chat(
        self,
        chat,
        max_tokens = -1,
        output_structure = None,
        stream = False,
        **kwargs,
    ) -> str:
        if output_structure is not None:
            output = output_structure.model_json_schema()['properties']
            for key in output.keys():
                output[key] = "This is a structured chat output"
            
        else:
            output = "This is unstructured chat output"
            
        return output
        
    async def a_run_chat(
        self,
        chat,
        max_tokens = -1,
        output_structure = None,
        stream = False,
        **kwargs,
    ) -> str:
        if output_structure is not None:
            output = output_structure.model_json_schema()['properties']
            for key in output.keys():
                output[key] = "This is a structured chat output"
            
        else:
            output = "This is unstructured chat output"  
            
        return output  
        
    def __call__(self, *args, **kwargs):
        return "direct LLM completion call"

    def num_chat_tokens(self, *args, **kwargs):
        return 100

    def tokenize(self, *args, **kwargs):
        return [1,2,3]

    def untokenize(self, *args, **kwargs):
        return "hello world"