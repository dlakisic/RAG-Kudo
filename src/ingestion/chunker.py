"""
Module de découpage sémantique de documents.
Divise les documents en chunks optimaux pour le RAG.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

from config import settings


@dataclass
class ChunkMetadata:
    """Métadonnées enrichies pour un chunk."""
    source_file: str
    chunk_id: int
    section: Optional[str] = None
    category: Optional[str] = None
    importance: Optional[str] = None
    article_reference: Optional[str] = None


class SemanticChunker:
    """Découpe sémantique de documents avec métadonnées enrichies."""

    def __init__(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        embed_model: Optional[OpenAIEmbedding] = None,
    ):
        """
        Initialise le chunker sémantique.

        Args:
            buffer_size: Nombre de phrases à grouper avant de calculer la similarité
            breakpoint_percentile_threshold: Seuil percentile pour détecter les ruptures sémantiques
            embed_model: Modèle d'embeddings (par défaut: OpenAI text-embedding-3-small)
        """
        self.embed_model = embed_model or OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=self.embed_model,
        )

        logger.info(
            f"SemanticChunker initialisé avec buffer_size={buffer_size}, "
            f"threshold={breakpoint_percentile_threshold}"
        )

    def chunk_document(
        self,
        doc_data: Dict,
        metadata_enrichment: Optional[Dict] = None
    ) -> List[TextNode]:
        """
        Découpe un document en chunks sémantiques.

        Args:
            doc_data: Données du document traité par Docling
            metadata_enrichment: Métadonnées supplémentaires à ajouter

        Returns:
            Liste de TextNode avec métadonnées enrichies
        """
        logger.info(f"Découpage du document: {doc_data.get('file_name')}")

        document = Document(
            text=doc_data.get("content", ""),
            metadata={
                "source_file": doc_data.get("source_file"),
                "file_name": doc_data.get("file_name"),
                "num_pages": doc_data.get("metadata", {}).get("num_pages"),
                **(metadata_enrichment or {}),
            },
        )

        nodes = self.splitter.get_nodes_from_documents([document])

        enriched_nodes = []
        for idx, node in enumerate(nodes):
            section_info = self._detect_section(node.text, doc_data.get("structure", []))

            node.metadata.update({
                "chunk_id": idx,
                "section": section_info.get("section"),
                "category": section_info.get("category"),
                "article_reference": section_info.get("article_ref"),
            })

            enriched_nodes.append(node)

        logger.success(f"Document découpé en {len(enriched_nodes)} chunks")
        return enriched_nodes

    def _detect_section(self, chunk_text: str, structure: List[Dict]) -> Dict:
        """
        Détecte la section et catégorie d'un chunk basé sur la structure du document.

        Args:
            chunk_text: Texte du chunk
            structure: Structure hiérarchique du document

        Returns:
            Dictionnaire avec section, catégorie, référence article
        """
        section_info = {
            "section": None,
            "category": None,
            "article_ref": None,
        }

        for item in structure:
            if item.get("type") in ["heading", "title"]:
                if item.get("text", "").lower() in chunk_text.lower():
                    section_info["section"] = item.get("text")

                    text_lower = item.get("text", "").lower()
                    if any(kw in text_lower for kw in ["technique", "coup", "frappe"]):
                        section_info["category"] = "techniques_autorisees"
                    elif any(kw in text_lower for kw in ["sanction", "faute", "interdit"]):
                        section_info["category"] = "sanctions"
                    elif any(kw in text_lower for kw in ["point", "score", "victoire"]):
                        section_info["category"] = "scoring"
                    elif any(kw in text_lower for kw in ["équipement", "tenue", "protection"]):
                        section_info["category"] = "equipement"
                    else:
                        section_info["category"] = "regles_generales"

                    break

        import re
        article_match = re.search(r"article\s+(\d+\.?\d*)", chunk_text.lower())
        if article_match:
            section_info["article_ref"] = f"Article {article_match.group(1)}"

        return section_info

    def chunk_multiple_documents(
        self,
        documents: List[Dict],
        metadata_enrichment: Optional[Dict] = None
    ) -> List[TextNode]:
        """
        Découpe plusieurs documents.

        Args:
            documents: Liste de documents traités
            metadata_enrichment: Métadonnées globales à ajouter

        Returns:
            Liste de tous les nodes
        """
        all_nodes = []

        for doc_data in documents:
            try:
                nodes = self.chunk_document(doc_data, metadata_enrichment)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(
                    f"Erreur lors du découpage de {doc_data.get('file_name')}: {e}"
                )
                continue

        logger.info(f"Total: {len(all_nodes)} chunks créés depuis {len(documents)} documents")
        return all_nodes


def main():
    """Fonction de test du module."""
    import json
    from pathlib import Path

    processed_file = Path("data/processed/exemple_processed.json")

    if processed_file.exists():
        with open(processed_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        chunker = SemanticChunker()
        nodes = chunker.chunk_document(doc_data)

        for i, node in enumerate(nodes[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"Section: {node.metadata.get('section')}")
            print(f"Catégorie: {node.metadata.get('category')}")
            print(f"Texte: {node.text[:200]}...")
    else:
        logger.warning(f"Fichier {processed_file} introuvable")


if __name__ == "__main__":
    main()
