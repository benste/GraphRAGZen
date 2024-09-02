from typing import List

from pydantic import BaseModel


class ExtractedCategories(BaseModel):
    categories: List[str]
