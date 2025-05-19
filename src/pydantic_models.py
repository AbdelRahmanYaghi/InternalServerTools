from pydantic import BaseModel
from typing import Optional, Any

class JSONTextEmbeddingRequest(BaseModel):
    text: list[str]
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs: Optional[dict] = {} 
    