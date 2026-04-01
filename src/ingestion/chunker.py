"""
Module de découpage sémantique de documents.
Divise les documents en chunks optimaux pour le RAG.
"""

from typing import List, Dict, Optional
import re
from loguru import logger

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document, TextNode

from config import settings
from src.utils.embeddings import build_embedding_model

MAX_CHUNK_SIZE = 1500


class SemanticChunker:
    """Découpe sémantique de documents avec métadonnées enrichies."""

    def __init__(
        self,
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        embed_model=None,
        max_chunk_size: int = MAX_CHUNK_SIZE,
    ):
        """
        Initialise le chunker sémantique.

        Args:
            buffer_size: Nombre de phrases à grouper avant de calculer la similarité
            breakpoint_percentile_threshold: Seuil percentile pour détecter les ruptures sémantiques
            embed_model: Modèle d'embeddings (par défaut: OpenAI text-embedding-3-small)
            max_chunk_size: Taille max d'un chunk en caractères avant re-découpage
        """
        self.embed_model = embed_model or build_embedding_model()
        self.max_chunk_size = max_chunk_size

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=self.embed_model,
        )

        logger.info(
            f"SemanticChunker initialisé avec buffer_size={buffer_size}, "
            f"threshold={breakpoint_percentile_threshold}, "
            f"max_chunk_size={max_chunk_size}"
        )

    def _is_table_of_contents(self, text: str) -> bool:
        """
        Détecte si un chunk est un sommaire/table des matières.

        Args:
            text: Texte du chunk

        Returns:
            True si c'est un sommaire
        """
        text_lower = text.lower()

        # Indicateurs de sommaire
        indicators = [
            r'\bsommaire\b',
            r'\btable\s+des\s+mati[èe]res\b',
            r'\btable\s+of\s+contents\b',
            r'page\s+\d+',
            r'\.\.\.\.\.\.',  # Points de suite typiques des sommaires
            r'……………',  # Points unicode
        ]

        indicator_count = sum(1 for pattern in indicators if re.search(pattern, text_lower))

        # Si 2+ indicateurs ou beaucoup de "page X"
        page_count = len(re.findall(r'page\s+\d+', text_lower))

        return indicator_count >= 2 or page_count >= 3

    def _is_procedural_content(self, text: str) -> bool:
        """
        Détecte si un chunk contient uniquement du contenu procédural
        (non pertinent pour les questions sur les règles).

        Args:
            text: Texte du chunk

        Returns:
            True si c'est du contenu procédural à exclure
        """
        text_lower = text.lower()

        # Sections procédurales à exclure
        procedural_patterns = [
            r"arrivée\s+des\s+arbitres",
            r"salut\s+au\s+tatami",
            r"shomen\s+ni\s+rei",
            r"otagani\s+rei",
            r"positionnement\s+des\s+arbitres",
            r"gestes?\s+d[''']arbitrage",  # Sauf si suivi de règles
            r"protocole\s+de\s+salut",
        ]

        # Si chunk court et contient un pattern procédural
        if len(text) < 200 and any(re.search(pattern, text_lower) for pattern in procedural_patterns):
            # Vérifier qu'il ne contient pas de règles importantes
            rule_keywords = ["interdit", "autoris", "point", "sanction", "durée", "équipement"]
            has_rules = any(kw in text_lower for kw in rule_keywords)
            return not has_rules

        return False

    def _extract_age_category(self, text: str) -> Optional[str]:
        """
        Extrait la catégorie d'âge mentionnée dans le texte.

        Args:
            text: Texte du chunk

        Returns:
            Catégorie d'âge ("U13", "U16", "U19", "adulte") ou None
        """
        patterns = {
            "U13": r'\bU\s*13\b',
            "U16": r'\bU\s*16\b',
            "U19": r'\bU\s*19\b',
            "adulte": r'\badultes?\b',
        }

        for category, pattern in patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return category

        return None

    def _detect_language(self, text: str) -> str:
        """
        Détecte grossièrement la langue dominante du chunk.

        Returns:
            Code langue simplifié: "ru", "fr", "en", "unknown"
        """
        cyrillic_chars = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
        latin_chars = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))

        if cyrillic_chars > max(20, latin_chars):
            return "ru"

        text_lower = text.lower()
        french_markers = (" le ", " la ", " les ", " des ", " est ", " sont ", " pour ")
        english_markers = (" the ", " and ", " is ", " are ", " with ", " for ")

        fr_score = sum(text_lower.count(m) for m in french_markers)
        en_score = sum(text_lower.count(m) for m in english_markers)

        if fr_score > en_score and fr_score > 0:
            return "fr"
        if en_score > fr_score and en_score > 0:
            return "en"
        if latin_chars > 0:
            return "en"

        return "unknown"

    def _detect_rule_type(self, text: str) -> Optional[str]:
        """
        Détecte le type de règle dans le texte.

        Args:
            text: Texte du chunk

        Returns:
            Type de règle ou None
        """
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["interdit", "interdiction", "prohibited", "forbidden"]):
            return "interdiction"
        elif any(kw in text_lower for kw in ["autoris", "permis", "allowed"]):
            return "autorisation"
        elif any(kw in text_lower for kw in ["koka", "yuko", "waza-ari", "ippon"]):
            return "scoring"
        elif any(kw in text_lower for kw in ["durée", "temps", "duration", "time"]):
            return "duree"
        elif any(kw in text_lower for kw in ["sanction", "faute", "penalty"]):
            return "sanction"

        return None

    def _has_exception(self, text: str) -> bool:
        """
        Détecte si le texte contient une exception ou condition.

        Args:
            text: Texte du chunk

        Returns:
            True si contient une exception
        """
        exception_keywords = [
            r'\bsauf\b',
            r'\bexcept\b',
            r'\bunless\b',
            r'\bsi\s+cela\s+est\s+spécifié\b',
            r'\bà\s+condition\b',
            r'\bwhen\b',
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in exception_keywords)

    def _split_oversized_node(self, node: TextNode) -> List[TextNode]:
        """
        Re-découpe un node trop long en sous-chunks sur des frontières naturelles.

        Coupe de préférence sur : lignes de tableau, headings markdown, doubles
        sauts de ligne, puis sauts de ligne simples en dernier recours.

        Args:
            node: Node dépassant max_chunk_size

        Returns:
            Liste de sous-nodes
        """
        text = node.text
        if len(text) <= self.max_chunk_size:
            return [node]

        split_patterns = [
            r'\n\|[^\n]+\|\n',   # ligne de tableau markdown
            r'\n#{1,4} ',        # heading markdown
            r'\n\n',             # double saut de ligne
            r'\n',               # saut de ligne simple
        ]

        chunks = [text]
        for pattern in split_patterns:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) <= self.max_chunk_size:
                    new_chunks.append(chunk)
                    continue
                parts = re.split(f'({pattern})', chunk)
                current = ""
                for part in parts:
                    if len(current) + len(part) > self.max_chunk_size and current.strip():
                        new_chunks.append(current.strip())
                        current = part
                    else:
                        current += part
                if current.strip():
                    new_chunks.append(current.strip())
            chunks = new_chunks

        sub_nodes = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            sub_node = TextNode(
                text=chunk_text,
                metadata={**node.metadata},
            )
            sub_nodes.append(sub_node)

        logger.info(
            f"Chunk de {len(text)} chars re-découpé en {len(sub_nodes)} sous-chunks"
        )
        return sub_nodes

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

        raw_nodes = self.splitter.get_nodes_from_documents([document])

        # Re-découper les chunks trop longs
        nodes = []
        for node in raw_nodes:
            if len(node.text) > self.max_chunk_size:
                nodes.extend(self._split_oversized_node(node))
            else:
                nodes.append(node)

        enriched_nodes = []
        filtered_count = 0

        for idx, node in enumerate(nodes):
            # Filtrage Phase 1: Exclure sommaires et contenu procédural
            if self._is_table_of_contents(node.text):
                logger.debug(f"Chunk {idx} filtré: sommaire détecté")
                filtered_count += 1
                continue

            is_procedural = self._is_procedural_content(node.text)
            if is_procedural:
                logger.debug(f"Chunk {idx} filtré: contenu procédural")
                filtered_count += 1
                continue

            # Enrichissement des métadonnées
            section_info = self._detect_section(node.text, doc_data.get("structure", []))

            node.metadata.update({
                "chunk_id": idx,
                "section": section_info.get("section"),
                "category": section_info.get("category"),
                "article_reference": section_info.get("article_ref"),
                # Nouvelles métadonnées Phase 1
                "age_category": self._extract_age_category(node.text),
                "language": self._detect_language(node.text),
                "rule_type": self._detect_rule_type(node.text),
                "has_exception": self._has_exception(node.text),
                "is_procedural": False,  # Déjà filtré
            })

            enriched_nodes.append(node)

        logger.success(
            f"Document découpé: {len(enriched_nodes)} chunks gardés, "
            f"{filtered_count} chunks filtrés"
        )
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
