"""
Module de reformulation de requêtes pour améliorer le recall.
Génère plusieurs variations d'une requête utilisateur pour augmenter les chances
de trouver des documents pertinents.
"""

from typing import List, Optional, TYPE_CHECKING
from loguru import logger

from llama_index.core.llms import ChatMessage, MessageRole

if TYPE_CHECKING:
    from src.generation.llm_manager import LLMManager


class QueryReformulator:
    """
    Reformule les requêtes utilisateur pour améliorer la recherche.

    Stratégies:
    1. Reformulation par le LLM (Query2Query)
    2. Décomposition en sous-questions
    """

    def __init__(
        self,
        llm_manager: Optional["LLMManager"] = None,
        use_llm_reformulation: bool = True,
        num_variations: int = 2,
    ):
        """
        Initialise le reformulateur.

        Args:
            llm_manager: Gestionnaire LLM pour la reformulation
            use_llm_reformulation: Utiliser le LLM pour reformuler
            num_variations: Nombre de variations à générer
        """
        if llm_manager is None:
            from src.generation.llm_manager import LLMManager
            # Reformulation deterministe pour améliorer la reproductibilite des runs d'evaluation.
            llm_manager = LLMManager(temperature=0.0)

        self.llm_manager = llm_manager
        self.use_llm_reformulation = use_llm_reformulation
        self.num_variations = num_variations
        self._cache: dict[str, List[str]] = {}

        logger.info(
            f"QueryReformulator initialisé avec {num_variations} variations LLM"
        )

    def reformulate(
        self,
        query: str,
        include_original: bool = True,
    ) -> List[str]:
        """
        Génère plusieurs variations d'une requête via LLM.

        Args:
            query: Requête originale
            include_original: Inclure la requête originale dans les résultats

        Returns:
            Liste de requêtes reformulées
        """
        logger.info(f"Reformulation de: '{query}'")
        cache_key = query.strip().lower()
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Reformulation récupérée depuis le cache")
            return list(cached)

        queries = []

        if include_original:
            queries.append(query)

        if self.use_llm_reformulation:
            llm_variations = self._llm_reformulate(query)
            queries.extend(llm_variations[:self.num_variations])

        unique_queries = []
        seen = set()
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        logger.info(f"Généré {len(unique_queries)} variations de requêtes")
        for i, q in enumerate(unique_queries, 1):
            logger.debug(f"  {i}. {q}")

        self._cache[cache_key] = list(unique_queries)
        return unique_queries

    def _llm_reformulate(self, query: str) -> List[str]:
        """
        Utilise le LLM pour générer des reformulations intelligentes.

        Args:
            query: Requête originale

        Returns:
            Liste de requêtes reformulées par le LLM
        """
        try:
            prompt = f"""Question originale: "{query}"

Génère {self.num_variations} reformulations de cette question qui:
1. Gardent le même sens dans le contexte du KUDO (art martial japonais / sport de combat)
2. Utilisent des termes synonymes ou la terminologie officielle KIF
3. Restent courtes et précises (max 15 mots)
4. Peuvent inclure des termes techniques en anglais ou en russe si cela améliore la recherche documentaire

IMPORTANT: Le Kudo est un art martial. Les termes comme "frappes", "coups", "techniques" font référence aux TECHNIQUES DE COMBAT, pas à autre chose.

Format: Une reformulation par ligne, sans numérotation.

Reformulations:"""

            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="""Tu es un assistant spécialisé en Kudo (Daido Juku), un art martial japonais combinant karaté, judo et boxe.
Ton rôle est de reformuler des questions sur le règlement d'arbitrage Kudo.
IMPORTANT: Reste TOUJOURS dans le contexte des sports de combat. Ne change JAMAIS le sens de la question.""",
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=prompt,
                ),
            ]

            response = self.llm_manager.chat(messages)

            variations = [
                line.strip()
                for line in response.strip().split('\n')
                if line.strip() and not line.strip().startswith(('-', '•', '*', '1.', '2.', '3.'))
            ]

            cleaned_variations = []
            for var in variations:
                import re
                cleaned = re.sub(r'^\d+\.\s*', '', var)
                cleaned = re.sub(r'^[-•*]\s*', '', cleaned)
                if cleaned and len(cleaned) > 10:
                    cleaned_variations.append(cleaned)

            logger.debug(f"LLM a généré {len(cleaned_variations)} reformulations")
            return cleaned_variations

        except Exception as e:
            logger.error(f"Erreur lors de la reformulation LLM: {e}")
            return []

    def decompose_query(self, query: str) -> List[str]:
        """
        Décompose une question complexe en sous-questions simples.

        Args:
            query: Question complexe

        Returns:
            Liste de sous-questions
        """
        try:
            prompt = f"""Décompose cette question complexe sur l'arbitrage Kudo en 2-3 sous-questions simples et spécifiques.

Question: "{query}"

Chaque sous-question doit:
- Porter sur un aspect précis de la question
- Être indépendante et compréhensible seule
- Utiliser la terminologie Kudo appropriée

Format: Une sous-question par ligne, sans numérotation.

Sous-questions:"""

            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="Tu es un expert en pédagogie de l'arbitrage Kudo.",
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=prompt,
                ),
            ]

            response = self.llm_manager.chat(messages)

            import re
            sub_queries = [
                re.sub(r'^\d+\.\s*', '', line.strip())
                for line in response.strip().split('\n')
                if line.strip() and len(line.strip()) > 10
            ]

            logger.info(f"Décomposé en {len(sub_queries)} sous-questions")
            return sub_queries

        except Exception as e:
            logger.error(f"Erreur lors de la décomposition: {e}")
            return [query]


def main():
    """Fonction de test du module."""
    reformulator = QueryReformulator(num_variations=2)

    test_queries = [
        "Comment sont attribués les points en Kudo ?",
        "Quelles sont les fautes graves ?",
        "Équipement obligatoire pour les combattants",
        "Que faire en cas de KO ?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Question originale: {query}")
        print(f"{'='*70}")

        # Test de reformulation
        variations = reformulator.reformulate(query)
        print("\n📝 Variations générées:")
        for i, var in enumerate(variations, 1):
            print(f"  {i}. {var}")

        # Test de décomposition
        if "et" in query or "comment" in query.lower():
            sub_queries = reformulator.decompose_query(query)
            print("\n🔍 Sous-questions:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")


if __name__ == "__main__":
    main()
