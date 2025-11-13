"""
Module de re-ranking pour améliorer la pertinence des résultats de recherche.
Utilise un CrossEncoder pour scorer les paires (query, document).
"""

from typing import List, Tuple
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("sentence-transformers non installé. Re-ranking désactivé.")

from llama_index.core.schema import NodeWithScore

from config import settings


class KudoReranker:
    """
    Re-ranker basé sur CrossEncoder pour améliorer la pertinence des résultats.

    Le re-ranking est une étape supplémentaire après la recherche vectorielle initiale
    qui utilise un modèle plus sophistiqué pour scorer les paires (query, document).
    """

    def __init__(self, model_name: str = None, use_gpu: bool = True):
        """
        Initialise le re-ranker.

        Args:
            model_name: Nom du modèle CrossEncoder à utiliser
            use_gpu: Utiliser le GPU si disponible
        """
        self.model_name = model_name or settings.reranker_model
        self.use_gpu = use_gpu
        self.model = None

        if not RERANKER_AVAILABLE:
            logger.warning("Re-ranking désactivé : sentence-transformers non disponible")
            return

        if not settings.use_reranking:
            logger.info("Re-ranking désactivé dans la configuration")
            return

        self._initialize_model()

    def _initialize_model(self):
        """Initialise le modèle CrossEncoder."""
        try:
            device = "cuda" if self.use_gpu else "cpu"

            logger.info(f"Chargement du CrossEncoder: {self.model_name} sur {device}")
            self.model = CrossEncoder(self.model_name, device=device)

            logger.success(f"Re-ranker initialisé avec {self.model_name}")

        except Exception as e:
            logger.error(f"Erreur lors du chargement du re-ranker: {e}")
            self.model = None

    def is_enabled(self) -> bool:
        """Vérifie si le re-ranking est activé et fonctionnel."""
        return self.model is not None and settings.use_reranking

    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: int = None
    ) -> List[NodeWithScore]:
        """
        Re-rank une liste de nodes en fonction de leur pertinence pour la query.

        Args:
            query: Question de l'utilisateur
            nodes: Liste de nodes à re-ranker
            top_k: Nombre de nodes à retourner (None = tous)

        Returns:
            Liste de nodes re-rankés par score décroissant
        """
        if not self.is_enabled():
            logger.debug("Re-ranking désactivé, retour des nodes originaux")
            return nodes

        if not nodes:
            return nodes

        try:
            logger.info(f"Re-ranking de {len(nodes)} nodes pour: '{query[:50]}...'")

            # Préparer les paires (query, document)
            pairs = [(query, node.node.get_content()) for node in nodes]

            # Obtenir les scores du CrossEncoder
            scores = self.model.predict(pairs)

            # Créer de nouveaux nodes avec les scores mis à jour
            reranked_nodes = []
            for node, score in zip(nodes, scores):
                # Créer un nouveau NodeWithScore avec le score du re-ranker
                reranked_node = NodeWithScore(
                    node=node.node,
                    score=float(score)
                )
                reranked_nodes.append(reranked_node)

            # Trier par score décroissant
            reranked_nodes.sort(key=lambda x: x.score, reverse=True)

            # Limiter au top_k si spécifié
            if top_k:
                reranked_nodes = reranked_nodes[:top_k]

            # Logging des changements
            original_top = nodes[0].node.get_content()[:100] if nodes else ""
            reranked_top = reranked_nodes[0].node.get_content()[:100] if reranked_nodes else ""

            if original_top != reranked_top:
                logger.info("Re-ranking a modifié l'ordre des résultats")
                logger.debug(f"Original top: {original_top}")
                logger.debug(f"Reranked top: {reranked_top}")

            logger.success(f"Re-ranking terminé. Retour de {len(reranked_nodes)} nodes")

            return reranked_nodes

        except Exception as e:
            logger.error(f"Erreur lors du re-ranking: {e}")
            logger.warning("Retour des nodes originaux sans re-ranking")
            return nodes

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Score une liste de paires (query, document).

        Args:
            pairs: Liste de tuples (query, document)

        Returns:
            Liste de scores de pertinence
        """
        if not self.is_enabled():
            return [0.0] * len(pairs)

        try:
            scores = self.model.predict(pairs)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            logger.error(f"Erreur lors du scoring: {e}")
            return [0.0] * len(pairs)
