"""
Retriever personnalisé pour le système RAG Kudo.
Implémente la recherche hybride et le re-ranking.
"""

from typing import List, Optional, Dict
from loguru import logger

from llama_index.core import VectorStoreIndex, Settings as LlamaSettings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SimilarityPostprocessor

from config import settings
from src.retrieval.reranker import KudoReranker
from src.retrieval.query_reformulator import QueryReformulator


class KudoRetriever:
    """Retriever optimisé pour les règles d'arbitrage Kudo."""

    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_reranking: Optional[bool] = None,
        use_query_reformulation: bool = True,
        metadata_filters: Optional[Dict] = None,
    ):
        """
        Initialise le retriever.

        Args:
            index: Index vectoriel
            top_k: Nombre de documents à récupérer
            similarity_threshold: Seuil de similarité minimum
            use_reranking: Utiliser le re-ranking
            use_query_reformulation: Utiliser la reformulation de requêtes
            metadata_filters: Filtres de métadonnées
        """
        self.index = index
        self.top_k = top_k or settings.top_k
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        self.use_reranking = use_reranking or settings.use_reranking
        self.use_query_reformulation = use_query_reformulation
        self.metadata_filters = metadata_filters or {}

        self.base_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k * 3,
            callback_manager=LlamaSettings.callback_manager,
        )

        self.similarity_postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_threshold
        )

        self.reranker = KudoReranker() if self.use_reranking else None

        self.query_reformulator = QueryReformulator(num_variations=2) if self.use_query_reformulation else None

        logger.info(
            f"KudoRetriever initialisé avec top_k={self.top_k}, "
            f"threshold={self.similarity_threshold}, "
            f"reranking={self.use_reranking}, "
            f"query_reformulation={self.use_query_reformulation}"
        )

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        Récupère les documents pertinents pour une requête.

        Args:
            query: Question de l'utilisateur

        Returns:
            Liste de nodes avec scores de pertinence
        """
        logger.info(f"Recherche pour: {query}")

        if self.use_query_reformulation and self.query_reformulator:
            query_variations = self.query_reformulator.reformulate(query)
            logger.info(f"Généré {len(query_variations)} variations de requêtes")

            all_nodes = []
            for i, q_var in enumerate(query_variations, 1):
                logger.debug(f"Recherche pour variation {i}: {q_var[:50]}...")
                query_bundle = QueryBundle(query_str=q_var)
                variation_nodes = self.base_retriever.retrieve(query_bundle)

                filtered_nodes = self.similarity_postprocessor.postprocess_nodes(variation_nodes)
                logger.debug(f"Variation {i}: {len(variation_nodes)} → {len(filtered_nodes)} après filtrage")

                all_nodes.append(filtered_nodes)

            nodes = self._fuse_results(all_nodes)
            logger.info(f"Fusion: {len(nodes)} nodes uniques après RRF")

        else:
            enhanced_query = self._enhance_query(query)
            query_bundle = QueryBundle(query_str=enhanced_query)
            nodes = self.base_retriever.retrieve(query_bundle)
            logger.info(f"Récupéré {len(nodes)} nodes initiaux")

            nodes = self.similarity_postprocessor.postprocess_nodes(nodes)
            logger.info(f"{len(nodes)} nodes après filtrage de similarité")

        if self.metadata_filters:
            nodes = self._filter_by_metadata(nodes, self.metadata_filters)
            logger.info(f"{len(nodes)} nodes après filtrage de métadonnées")

        if self.use_reranking and len(nodes) > 1:
            nodes = self._rerank_nodes(query, nodes)
            logger.info(f"Re-ranking appliqué")

        nodes = nodes[: self.top_k]

        logger.success(f"Retourné {len(nodes)} nodes finaux")
        return nodes

    def _enhance_query(self, query: str) -> str:
        """
        Améliore la requête avec des termes et contexte supplémentaires.

        Args:
            query: Requête originale

        Returns:
            Requête améliorée
        """
        query_lower = query.lower()

        expansions = {
            "point": ["point", "score", "marquage"],
            "faute": ["faute", "sanction", "pénalité", "infraction"],
            "ko": ["ko", "knock-out", "mise hors de combat"],
            "coup": ["coup", "technique", "frappe", "attaque"],
            "protection": ["protection", "équipement", "casque", "gants"],
            "arbitre": ["arbitre", "officiel", "jugement"],
        }

        enhanced_terms = []
        for key, synonyms in expansions.items():
            if key in query_lower:
                enhanced_terms.extend(synonyms[:2])

        if enhanced_terms:
            enhanced_query = f"{query} {' '.join(enhanced_terms)}"
            logger.debug(f"Requête améliorée: {enhanced_query}")
            return enhanced_query

        return query

    def _filter_by_metadata(
        self, nodes: List[NodeWithScore], filters: Dict
    ) -> List[NodeWithScore]:
        """
        Filtre les nodes par métadonnées.

        Args:
            nodes: Liste de nodes
            filters: Filtres à appliquer (ex: {"category": "sanctions"})

        Returns:
            Nodes filtrés
        """
        filtered_nodes = []

        for node in nodes:
            metadata = node.node.metadata
            match = all(
                metadata.get(key) == value for key, value in filters.items()
            )
            if match:
                filtered_nodes.append(node)

        return filtered_nodes

    def _rerank_nodes(
        self, query: str, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Re-classe les nodes avec un modèle de cross-encoding.

        Args:
            query: Requête originale
            nodes: Nodes à re-classer

        Returns:
            Nodes re-classés
        """
        if self.reranker and self.reranker.is_enabled():
            return self.reranker.rerank(query, nodes, top_k=None)

        logger.debug("Re-ranker non disponible, retour des nodes originaux")
        return nodes

    def _fuse_results(
        self, all_nodes: List[List[NodeWithScore]], k: int = 60
    ) -> List[NodeWithScore]:
        """
        Fusionne les résultats de plusieurs requêtes avec Reciprocal Rank Fusion (RRF).

        RRF est une technique qui combine des listes de résultats classés en fonction
        de leur position dans chaque liste. Score RRF = sum(1 / (k + rank)) pour chaque liste.

        Args:
            all_nodes: Liste de listes de nodes (une par variation de requête)
            k: Constante RRF (généralement 60, contrôle l'importance du rang)

        Returns:
            Nodes fusionnés et triés par score RRF
        """
        from collections import defaultdict

        rrf_scores = defaultdict(float)
        node_map = {}

        for node_list in all_nodes:
            for rank, node in enumerate(node_list, start=1):
                node_id = node.node.node_id

                rrf_score = 1.0 / (k + rank)
                rrf_scores[node_id] += rrf_score

                if node_id not in node_map:
                    node_map[node_id] = node

        fused_nodes = []
        for node_id, rrf_score in rrf_scores.items():
            node = node_map[node_id]
            fused_node = NodeWithScore(
                node=node.node,
                score=rrf_score
            )
            fused_nodes.append(fused_node)

        fused_nodes.sort(key=lambda x: x.score, reverse=True)

        logger.debug(f"RRF: fusionné {len(all_nodes)} listes en {len(fused_nodes)} nodes uniques")

        return fused_nodes

    def retrieve_by_category(
        self, query: str, category: str
    ) -> List[NodeWithScore]:
        """
        Récupère des documents d'une catégorie spécifique.

        Args:
            query: Question
            category: Catégorie (ex: "sanctions", "techniques_autorisees")

        Returns:
            Nodes de la catégorie
        """
        original_filters = self.metadata_filters.copy()

        self.metadata_filters["category"] = category

        nodes = self.retrieve(query)

        self.metadata_filters = original_filters

        return nodes

    def retrieve_with_context(
        self, query: str, previous_context: Optional[List[str]] = None
    ) -> List[NodeWithScore]:
        """
        Récupère avec prise en compte du contexte conversationnel.

        Args:
            query: Question actuelle
            previous_context: Questions/réponses précédentes

        Returns:
            Nodes pertinents
        """
        if previous_context:
            context_str = " ".join(previous_context[-3:])
            enriched_query = f"{context_str} {query}"
            logger.debug(f"Requête avec contexte: {enriched_query}")
            return self.retrieve(enriched_query)

        return self.retrieve(query)


def main():
    """Fonction de test du module."""
    from src.retrieval.vector_store import VectorStoreManager

    manager = VectorStoreManager()

    try:
        index = manager.load_index()

        retriever = KudoRetriever(index=index)

        test_queries = [
            "Quelles sont les techniques de frappe autorisées ?",
            "Comment sont attribués les points ?",
            "Quelles sanctions peuvent être données ?",
        ]

        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Question: {query}")
            print(f"{'='*60}")

            nodes = retriever.retrieve(query)

            for i, node in enumerate(nodes, 1):
                print(f"\n[Résultat {i}] Score: {node.score:.3f}")
                print(f"Catégorie: {node.node.metadata.get('category', 'N/A')}")
                print(f"Section: {node.node.metadata.get('section', 'N/A')}")
                print(f"Texte: {node.node.get_content()[:200]}...")

    except Exception as e:
        logger.error(f"Erreur: {e}")
        print("\nL'index n'existe pas encore. Veuillez d'abord créer l'index.")


if __name__ == "__main__":
    main()
