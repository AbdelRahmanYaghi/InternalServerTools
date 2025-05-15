from pydantic import BaseModel
from typing import Optional, Any

class JSONTextEmbeddingRequest(BaseModel):
    text: list[str]
    embedding_model: str
    model_kwargs: Optional[dict] = {} 
    