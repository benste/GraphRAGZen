from typing import List

from pydantic import BaseModel, Field


class ExtractedNode(BaseModel):
    type: str = "node"
    name: str
    category: str
    description: str = Field(
        description="Short explanation as to why you think the source node and the target node are related to each other"  # noqa: E501
    )


class ExtractedEdge(BaseModel):
    type: str = "edge"
    source: str
    target: str
    description: str = Field(
        description="Short explanation as to why you think the source node and the target node are related to each other"  # noqa: E501
    )
    weight: float


class ExtractedEntities(BaseModel):
    extracted_nodes: List[ExtractedNode]
    extracted_edges: List[ExtractedEdge]
