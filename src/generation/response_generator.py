"""
Générateur de réponses pour le système RAG Kudo.
Intègre retrieval et génération avec prompts optimisés pour la formation.
"""

from typing import List, Optional, Dict
from loguru import logger

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole

from src.retrieval.retriever import KudoRetriever
from src.generation.llm_manager import LLMManager
from src.observability.langfuse_manager import get_langfuse_manager
from config import settings


# Prompts optimisés pour la formation d'arbitres
SYSTEM_PROMPT = """Tu es un formateur expert en arbitrage de Kudo (Daido Juku).

OBJECTIF : Répondre à la question en te basant sur le contexte fourni. Le contexte peut être en français, anglais ou russe — extrais l'information pertinente et réponds toujours en français.

RÈGLES :
1. Base-toi UNIQUEMENT sur les informations présentes dans le contexte
2. Si le contexte contient l'information (même dans une autre langue), RÉPONDS — ne dis pas que l'info est absente
3. Ne dis "Cette information n'est pas présente dans les documents fournis" que si AUCUN extrait ne traite du sujet de la question
4. N'invente pas d'articles, de chiffres ou de règles absents du contexte

FORMAT :
1. Cite le passage pertinent du contexte (traduis-le en français si nécessaire)
2. Explique brièvement
3. Indique la référence (Article X, Section Y) si disponible

Réponds en français, de manière concise."""

QA_PROMPT_TEMPLATE = """CONTEXTE DU RÈGLEMENT OFFICIEL :
{context_str}

---

QUESTION DE L'ARBITRE EN FORMATION :
{query_str}

---

INSTRUCTIONS :
- Réponds en te basant sur le contexte ci-dessus (même s'il est en anglais ou russe, réponds en français)
- Si le contexte contient des éléments de réponse, utilise-les — ne refuse pas de répondre
- Cite les règles et références du règlement (article, section) si disponibles
- Ne dis que l'information est absente que si AUCUN extrait ne traite du sujet

Réponse :"""

class KudoResponseGenerator:
    """Générateur de réponses pour la formation d'arbitres Kudo."""

    def __init__(
        self,
        index: VectorStoreIndex,
        llm_manager: Optional[LLMManager] = None,
        retriever: Optional[KudoRetriever] = None,
    ):
        """
        Initialise le générateur de réponses.

        Args:
            index: Index vectoriel
            llm_manager: Gestionnaire LLM
            retriever: Retriever personnalisé
        """
        self.index = index
        self.llm_manager = llm_manager or LLMManager()
        self.retriever = retriever or KudoRetriever(index=index)

        self.langfuse_manager = get_langfuse_manager()
        if self.langfuse_manager.is_enabled():
            logger.info("LangFuse observabilité activée")

        logger.info("KudoResponseGenerator initialisé")

    async def generate_stream(
        self,
        question: str,
        include_sources: bool = True,
        conversation_history: Optional[List[Dict]] = None,
    ):
        """
        Génère une réponse en mode streaming pour Chainlit.

        Args:
            question: Question de l'utilisateur
            include_sources: Inclure les sources dans la réponse
            conversation_history: Historique de conversation

        Yields:
            Tuples (token, nodes) où token est le texte streamé et nodes sont les sources
        """
        logger.info(f"Génération de réponse en streaming pour: {question}")

        try:
            nodes = self._retrieve_nodes(question, conversation_history)
            messages = self._build_messages(question, nodes, conversation_history)

            for token in self.llm_manager.stream_chat(messages):
                yield token, nodes

        except Exception as e:
            logger.error(f"Erreur lors de la génération streaming: {e}")
            raise

    def generate(
        self,
        question: str,
        include_sources: bool = True,
        conversation_history: Optional[List[Dict]] = None,
        retrieved_nodes=None,
    ) -> Dict:
        """
        Génère une réponse complète à une question.

        Args:
            question: Question de l'utilisateur
            include_sources: Inclure les sources dans la réponse
            conversation_history: Historique de conversation
            retrieved_nodes: Nodes déjà récupérés (évite un retrieval supplémentaire)

        Returns:
            Dictionnaire avec la réponse et métadonnées
        """
        logger.info(f"Génération de réponse pour: {question}")

        try:
            nodes = retrieved_nodes if retrieved_nodes is not None else self._retrieve_nodes(
                question, conversation_history
            )
            messages = self._build_messages(question, nodes, conversation_history)
            response = self.llm_manager.chat(messages)

            result = {
                "question": question,
                "answer": str(response),
                "sources": [],
                "confidence": self._compute_confidence(nodes),
                "num_sources": len(nodes),
            }

            if include_sources and settings.enable_citations:
                result["sources"] = self._format_sources(nodes)

            logger.success("Réponse générée avec succès")
            return result

        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise

    def generate_with_examples(self, question: str) -> Dict:
        """
        Génère une réponse enrichie avec des exemples pratiques.

        Args:
            question: Question de l'utilisateur

        Returns:
            Réponse avec exemples
        """
        nodes = self.retriever.retrieve(question)

        enriched_prompt = f"""Question : {question}

En te basant sur le contexte fourni, réponds en incluant :
1. La règle officielle
2. Au moins 2 exemples concrets de situations d'arbitrage
3. Les erreurs courantes à éviter
4. Les références du règlement

Structure ta réponse clairement."""

        messages = self._build_messages(enriched_prompt, nodes)
        response = self.llm_manager.chat(messages)

        return {
            "question": question,
            "answer": str(response),
            "sources": self._format_sources(nodes),
        }

    def generate_quiz_question(self, category: Optional[str] = None) -> Dict:
        """
        Génère une question de quiz pour l'entraînement.

        Args:
            category: Catégorie de la question (ex: "sanctions")

        Returns:
            Question de quiz avec réponse
        """
        if category:
            context_query = f"règles de {category}"
        else:
            context_query = "règles d'arbitrage Kudo"

        nodes = self.retriever.retrieve(context_query)

        if not nodes:
            raise ValueError("Aucun contexte disponible pour générer une question")

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Basé sur ce contexte du règlement :
{nodes[0].node.get_content()}

Génère une question de quiz pour tester un arbitre en formation.
Format :
Question: [la question]
Réponse: [la réponse correcte]
Explication: [explication détaillée]""",
            ),
        ]

        quiz = self.llm_manager.chat(messages)

        return {
            "quiz": quiz,
            "source_section": nodes[0].node.metadata.get("section"),
            "category": nodes[0].node.metadata.get("category"),
        }

    def explain_decision(
        self, situation: str, decision: str
    ) -> Dict:
        """
        Explique une décision d'arbitrage dans une situation donnée.

        Args:
            situation: Description de la situation
            decision: Décision prise par l'arbitre

        Returns:
            Explication détaillée
        """
        query = f"{situation} {decision}"
        nodes = self.retriever.retrieve(query)

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"""Situation : {situation}
Décision : {decision}

Contexte règlementaire :
{self._format_context(nodes)}

Explique si cette décision est correcte et pourquoi. Base ton explication sur les règles officielles.""",
            ),
        ]

        explanation = self.llm_manager.chat(messages)

        return {
            "situation": situation,
            "decision": decision,
            "explanation": explanation,
            "relevant_rules": self._format_sources(nodes),
        }

    def _retrieve_nodes(
        self,
        question: str,
        conversation_history: Optional[List[Dict]] = None,
    ):
        if conversation_history:
            context = [msg["content"] for msg in conversation_history[-3:]]
            return self.retriever.retrieve_with_context(question, context)
        return self.retriever.retrieve(question)

    def _build_messages(
        self,
        question: str,
        nodes,
        conversation_history: Optional[List[Dict]] = None,
    ) -> List[ChatMessage]:
        context_str = self._format_context(nodes)
        user_prompt = QA_PROMPT_TEMPLATE.format(context_str=context_str, query_str=question)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]

        if conversation_history:
            for msg_dict in conversation_history[-6:]:
                role = MessageRole.USER if msg_dict["role"] == "user" else MessageRole.ASSISTANT
                messages.append(ChatMessage(role=role, content=msg_dict["content"]))

        messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))
        return messages

    def _compute_confidence(self, nodes) -> float:
        """
        Calcule un score de confiance basé sur les scores de similarité.

        Args:
            nodes: Nodes récupérés

        Returns:
            Score de confiance entre 0 et 1
        """
        if not nodes:
            return 0.0

        import math

        has_negative = any(node.score < 0 for node in nodes)

        if has_negative:
            # Scores de CrossEncoder dans [-10, +10], transformer avec sigmoid vers [0, 1]
            sigmoid_scores = [
                1.0 / (1.0 + math.exp(-node.score))
                for node in nodes
            ]
            avg_score = sum(sigmoid_scores) / len(sigmoid_scores)
        else:
            avg_score = sum(node.score for node in nodes) / len(nodes)

        return min(max(avg_score, 0.0), 1.0)

    def _format_sources(self, nodes) -> List[Dict]:
        """
        Formate les sources pour l'affichage.

        Args:
            nodes: Nodes sources

        Returns:
            Liste de sources formatées
        """
        sources = []
        for i, node in enumerate(nodes, 1):
            sources.append({
                "source_id": i,
                "file": node.node.metadata.get("file_name", "N/A"),
                "section": node.node.metadata.get("section", "N/A"),
                "category": node.node.metadata.get("category", "N/A"),
                "article_reference": node.node.metadata.get("article_reference", "N/A"),
                "relevance_score": round(node.score, 3),
                "excerpt": node.node.get_content()[:200] + "...",
            })
        return sources

    def _format_context(self, nodes) -> str:
        """
        Formate le contexte pour l'inclusion dans un prompt.

        Args:
            nodes: Nodes de contexte

        Returns:
            Contexte formaté
        """
        context_parts = []
        for i, node in enumerate(nodes, 1):
            section = node.node.metadata.get("section", "N/A")
            text = node.node.get_content()
            context_parts.append(f"[Source {i} - {section}]\n{text}\n")

        return "\n".join(context_parts)


def main():
    """Fonction de test du module."""
    from src.retrieval.vector_store import VectorStoreManager

    manager = VectorStoreManager()

    try:
        index = manager.load_index()

        generator = KudoResponseGenerator(index=index)

        test_questions = [
            "Quelles sont les techniques de frappe autorisées en Kudo ?",
            "Comment sont attribués les points dans un combat ?",
            "Que doit faire un arbitre en cas de faute ?",
        ]

        for question in test_questions:
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}\n")

            result = generator.generate(question)

            print(f"Réponse:\n{result['answer']}\n")
            print(f"Confiance: {result['confidence']:.2f}")
            print(f"Nombre de sources: {result['num_sources']}")

            if result["sources"]:
                print("\nSources:")
                for source in result["sources"][:2]:
                    print(f"  - {source['section']} ({source['category']})")
                    print(f"    Score: {source['relevance_score']}")

    except Exception as e:
        logger.error(f"Erreur: {e}")
        print("\nL'index n'existe pas encore. Veuillez d'abord créer l'index.")


if __name__ == "__main__":
    main()
