"""
Chunker avec embeddings locaux (sans OpenAI).
Utilise Sentence Transformers sur GPU.
"""

from typing import List, Dict, Optional
from loguru import logger

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.utils.gpu_utils import get_device


class LocalSemanticChunker:
    """Découpe sémantique avec embeddings locaux (GPU)."""

    def __init__(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialise le chunker avec embeddings locaux.

        Args:
            buffer_size: Nombre de phrases à grouper
            breakpoint_percentile_threshold: Seuil de rupture sémantique
            model_name: Modèle d'embeddings HuggingFace
        """
        device = get_device()

        logger.info(f"Chargement du modèle d'embeddings: {model_name}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device=str(device),
        )

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=self.embed_model,
        )

        logger.info(
            f"LocalSemanticChunker initialisé avec {model_name} sur {device}"
        )

    def chunk_document(
        self,
        doc_data: Dict,
        metadata_enrichment: Optional[Dict] = None
    ) -> List[TextNode]:
        """
        Découpe un document en chunks sémantiques.

        Args:
            doc_data: Données du document traité
            metadata_enrichment: Métadonnées supplémentaires

        Returns:
            Liste de TextNode
        """
        logger.info(f"Découpage du document: {doc_data.get('file_name')}")

        document = Document(
            text=doc_data.get("content", ""),
            metadata={
                "source_file": doc_data.get("source_file"),
                "file_name": doc_data.get("file_name"),
                "num_pages": doc_data.get("metadata", {}).get("num_pages"),
                "language": doc_data.get("language_hint", "unknown"),
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
        """Détecte la section et catégorie d'un chunk."""
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
                    elif any(kw in text_lower for kw in ["technique", "strike", "attack"]):
                        section_info["category"] = "techniques_autorisees"
                    elif any(kw in text_lower for kw in ["penalty", "foul", "violation"]):
                        section_info["category"] = "sanctions"
                    elif any(kw in text_lower for kw in ["scoring", "point", "victory"]):
                        section_info["category"] = "scoring"
                    elif any(kw in text_lower for kw in ["техник", "удар", "прием"]):
                        section_info["category"] = "techniques_autorisees"
                    elif any(kw in text_lower for kw in ["наказан", "нарушен", "штраф"]):
                        section_info["category"] = "sanctions"
                    elif any(kw in text_lower for kw in ["очк", "балл", "победа"]):
                        section_info["category"] = "scoring"

                    else:
                        section_info["category"] = "regles_generales"

                    break

        import re
        article_match = re.search(r"article\s+(\d+\.?\d*)", chunk_text.lower())
        if not article_match:
            article_match = re.search(r"статья\s+(\d+\.?\d*)", chunk_text.lower())
        if article_match:
            section_info["article_ref"] = f"Article {article_match.group(1)}"

        return section_info

    def chunk_multiple_documents(
        self,
        documents: List[Dict],
        metadata_enrichment: Optional[Dict] = None
    ) -> List[TextNode]:
        """Découpe plusieurs documents."""
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


if __name__ == "__main__":
    import json
    from pathlib import Path

    processed_file = Path("data/processed/01 L'ARBITRAGE AU KUDO_processed.json")

    if processed_file.exists():
        with open(processed_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)

        chunker = LocalSemanticChunker()
        nodes = chunker.chunk_document(doc_data)

        for i, node in enumerate(nodes[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"Section: {node.metadata.get('section')}")
            print(f"Catégorie: {node.metadata.get('category')}")
            print(f"Texte: {node.text[:200]}...")
