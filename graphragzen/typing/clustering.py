from pydantic import BaseModel

class ClusterConfig(BaseModel):
    """Config for loading local LLM"""
    max_comm_size: int = 10  # maximum number of nodes in one cluster


