"""
Fabrique de modeles d'embeddings selon la configuration.
"""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

from config import settings


def requires_openai_embeddings(model_name: str | None = None) -> bool:
    """Indique si la configuration d'embeddings necessite OPENAI_API_KEY."""
    embedding_model = model_name or settings.embedding_model
    return embedding_model.startswith("text-embedding-")


def build_embedding_model(model_name: str | None = None) -> BaseEmbedding:
    """Construit le modele d'embeddings adapte a la configuration."""
    embedding_model = model_name or settings.embedding_model

    if requires_openai_embeddings(embedding_model):
        return OpenAIEmbedding(
            model=embedding_model,
            api_key=settings.openai_api_key,
        )

    return HuggingFaceEmbedding(model_name=embedding_model)
