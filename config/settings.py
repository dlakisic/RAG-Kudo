"""
Configuration centrale du système RAG-Kudo.
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration globale de l'application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Chemins
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    raw_data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "processed")
    vectorstore_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "vectorstore")

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Modèles LLM
    llm_provider: Literal["openai", "anthropic"] = Field(default="openai")
    llm_model: str = Field(default="gpt-4-turbo-preview")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(default=1536)

    # Chunking
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    semantic_buffer_size: int = Field(default=1, gt=0)
    semantic_breakpoint_threshold: int = Field(default=95, ge=0, le=100)

    # Vector Store
    vectorstore_type: Literal["chroma", "qdrant", "pinecone"] = Field(default="chroma")
    collection_name: str = Field(default="kudo_arbitrage")

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_reranking: bool = Field(default=False)
    reranker_model: Optional[str] = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Generation
    max_tokens: int = Field(default=2000, gt=0)
    enable_citations: bool = Field(default=True)
    response_language: str = Field(default="fr")

    # Docling
    docling_extract_tables: bool = Field(default=True)
    docling_extract_images: bool = Field(default=False)
    docling_ocr_enabled: bool = Field(default=True)

    # GPU Configuration
    use_gpu: bool = Field(default=True)
    embedding_batch_size: int = Field(default=64, gt=0)
    ocr_batch_size: int = Field(default=4, gt=0)

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)

    # LangFuse Observability
    langfuse_enabled: bool = Field(default=False)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, gt=0, le=65535)
    api_reload: bool = Field(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Créer les répertoires s'ils n'existent pas
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


# Instance globale de configuration
settings = Settings()


def get_settings() -> Settings:
    """Retourne l'instance de configuration globale."""
    return settings
