from .constants import MODEL_PATHS 
from sentence_transformers import SentenceTransformer
import os 
import shutil

def get_downloaded_models() -> list[str]:
    out_dict = {}
    for modality in os.listdir(os.path.join(MODEL_PATHS)):
        out_dict[modality] = []
        model_providers = os.listdir(os.path.join(MODEL_PATHS, modality))
        for provider in model_providers:
            out_dict[modality].extend([provider + '/' + model for model in os.listdir(os.path.join(MODEL_PATHS, modality, provider))])
        return out_dict

def get_model(embedding_model: str, model_kwargs: dict):
    if embedding_model in get_downloaded_models()['text']:
         print(f"Model {embedding_model} already exists.")
         model = SentenceTransformer(os.path.join(MODEL_PATHS, "text", embedding_model), **model_kwargs)
    else:
        print(f"Model {embedding_model} not found. Downloading model commencing.") 
        model = SentenceTransformer(embedding_model, **model_kwargs)
        model.save(os.path.join(MODEL_PATHS, "text", embedding_model))
    
    return model

def delete_model(model_path):
    if os.path.exists(os.path.join(MODEL_PATHS, model_path)):
        shutil.rmtree(os.path.join(MODEL_PATHS, model_path))
        return f"Success. Model {model_path} removed successfully"
    else:
        return f"Failed. Model {model_path} not found"