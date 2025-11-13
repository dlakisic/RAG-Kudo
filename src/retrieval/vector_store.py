"""
Gestionnaire de base de données vectorielle.
Supporte ChromaDB, Qdrant et autres vector stores.
"""

from pathlib import Path
from typing import List, Optional
from loguru import logger

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

from config import settings


class VectorStoreManager:
    """Gestionnaire de base de données vectorielle."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialise le gestionnaire de vector store.

        Args:
            collection_name: Nom de la collection
            persist_dir: Répertoire de persistance
            embedding_model: Modèle d'embeddings
        """
        self.collection_name = collection_name or settings.collection_name
        self.persist_dir = persist_dir or settings.vectorstore_dir
        self.embedding_model_name = embedding_model or settings.embedding_model

        self.embed_model = OpenAIEmbedding(
            model=self.embedding_model_name,
            api_key=settings.openai_api_key,
        )

        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )

        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        self.index: Optional[VectorStoreIndex] = None

        logger.info(
            f"VectorStoreManager initialisé avec collection '{self.collection_name}' "
            f"dans {self.persist_dir}"
        )

    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """
        Crée un index à partir de nodes.

        Args:
            nodes: Liste de TextNode à indexer

        Returns:
            VectorStoreIndex créé
        """
        logger.info(f"Création de l'index avec {len(nodes)} nodes")

        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        logger.success(f"Index créé avec succès: {len(nodes)} nodes indexés")
        return self.index

    def load_index(self) -> VectorStoreIndex:
        """
        Charge un index existant depuis le stockage.

        Returns:
            VectorStoreIndex chargé
        """
        logger.info(f"Chargement de l'index depuis {self.persist_dir}")

        try:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model,
            )
            logger.success("Index chargé avec succès")
            return self.index

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'index: {e}")
            raise

    def add_nodes(self, nodes: List[TextNode]) -> None:
        """
        Ajoute des nodes à l'index existant.

        Args:
            nodes: Liste de TextNode à ajouter
        """
        if self.index is None:
            raise ValueError("Index non initialisé. Créez d'abord un index.")

        logger.info(f"Ajout de {len(nodes)} nodes à l'index")
        self.index.insert_nodes(nodes)
        logger.success(f"{len(nodes)} nodes ajoutés avec succès")

    def delete_collection(self) -> None:
        """Supprime la collection actuelle."""
        logger.warning(f"Suppression de la collection '{self.collection_name}'")
        self.chroma_client.delete_collection(name=self.collection_name)
        logger.success("Collection supprimée")

    def get_stats(self) -> dict:
        """
        Retourne les statistiques de la collection.

        Returns:
            Dictionnaire avec les statistiques
        """
        try:
            count = self.chroma_collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_dir": str(self.persist_dir),
                "embedding_model": self.embedding_model_name,
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stats: {e}")
            return {}

    def search_by_metadata(self, metadata_filter: dict, limit: int = 10) -> List[dict]:
        """
        Recherche par filtres de métadonnées.

        Args:
            metadata_filter: Filtres de métadonnées (ex: {"category": "sanctions"})
            limit: Nombre maximum de résultats

        Returns:
            Liste de documents correspondants
        """
        logger.info(f"Recherche avec filtres: {metadata_filter}")

        try:
            results = self.chroma_collection.query(
                query_texts=[""],
                where=metadata_filter,
                n_results=limit,
            )
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []


def main():
    """Fonction de test du module."""
    from llama_index.core.schema import TextNode

    test_nodes = [
        TextNode(
            text="Les techniques de frappe autorisées en Kudo incluent les coups de poing et de pied.",
            metadata={
                "source_file": "test.pdf",
                "category": "techniques_autorisees",
                "section": "Techniques autorisées",
            },
        ),
        TextNode(
            text="Les sanctions peuvent aller de l'avertissement à la disqualification.",
            metadata={
                "source_file": "test.pdf",
                "category": "sanctions",
                "section": "Système de sanctions",
            },
        ),
    ]

    manager = VectorStoreManager()
    index = manager.create_index(test_nodes)
    stats = manager.get_stats()
    print("\nStatistiques:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
