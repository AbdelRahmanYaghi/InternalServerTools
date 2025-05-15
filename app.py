from src.pydantic_models import JSONTextEmbeddingRequest
from src.helper import get_downloaded_models, get_model
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
    
@server.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs", include_in_schema=False)

@server.post("/embed_text")
def embed_text(request: JSONTextEmbeddingRequest) ->  list[list[float]]:
    model = get_model(request.embedding_model, request.model_kwargs)
    embedding = model.encode(request.text).tolist()
    return embedding

@server.get("/get_downloaded_models/")
def get_downloaded_text_models():
    return get_downloaded_models()
    
if __name__ == "__main__":
    __init__()
    uvicorn.run("app:server",host="0.0.0.0", port=8000)