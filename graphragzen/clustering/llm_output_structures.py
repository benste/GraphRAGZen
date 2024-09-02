from typing import List

from pydantic import BaseModel


class Finding(BaseModel):
    summary: str
    explanation: str


class ClusterDescription(BaseModel):
    title: str
    summary: str
    rating: float
    rating_explanation: str
    findings: List[Finding]
