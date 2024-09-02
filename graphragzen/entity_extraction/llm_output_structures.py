from typing import List

from pydantic import BaseModel


class ExtractedNode(BaseModel):
    type: str = "node"
    name: str
    category: str
    description: str


class ExtractedEdge(BaseModel):
    type: str = "edge"
    source: str
    target: str
    description: str
    weight: float


class ExtractedEntities(BaseModel):
    extracted_nodes: List[ExtractedNode]
    extracted_edges: List[ExtractedEdge]
