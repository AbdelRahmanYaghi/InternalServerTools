from src.pydantic_models import JSONTextEmbeddingRequest
from src.helper import get_downloaded_models, get_model, delete_model
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

server = FastAPI()

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def __init__():
    os.makedirs("models/text", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
@server.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@server.post("/embed_text")
def embed_text(request: JSONTextEmbeddingRequest) ->  list[list[float]]:
    model = get_model(request.embedding_model, request.model_kwargs)
    embedding = model.encode(request.text).tolist()
    return embedding

@server.get("/get_downloaded_models/")
def get_downloaded_models_():
    return get_downloaded_models()
    
@server.get("/delete_model/{model_path:path}")
def delete_model_(model_path: str):
    """
    Usage: ```{modality}/{model_provider}/{model_name}```
    
    example: ```delete_model("text/sentence-transformers/all-MiniLM-L6-v2")```
    """
    return delete_model(model_path)

if __name__ == "__main__":
    __init__()
    uvicorn.run("app:server",host="0.0.0.0", port=8000)