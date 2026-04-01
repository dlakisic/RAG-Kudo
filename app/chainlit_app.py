"""
Interface Chainlit pour le système RAG-Kudo.
Chat interactif avec affichage des sources et feedback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
from loguru import logger

from src.retrieval import VectorStoreManager
from src.generation import KudoResponseGenerator
from config import settings


@cl.on_chat_start
async def start():
    """Initialisation au démarrage du chat."""
    await cl.Message(
        content="""# 🥋 Bienvenue sur RAG-Kudo !

Je suis votre assistant pour la formation des arbitres de Kudo.

**Je peux vous aider avec :**
- 📖 Les règles d'arbitrage en Kudo
- ⚖️ Le système de scoring et de pénalités
- 🥊 Les techniques autorisées/interdites
- 👔 L'équipement réglementaire
- 🇫🇷 🇬🇧 🇷🇺 Questions en français, anglais ou russe

**Exemples de questions :**
- "Quelles sont les techniques de frappe autorisées en U16 ?"
- "Comment marque-t-on un ippon ?"
- "What is the required athlete's attire?"

Posez-moi vos questions sur l'arbitrage en Kudo ! 👇
""",
        author="Assistant"
    ).send()

    try:
        await cl.Message(content="⏳ Chargement du système RAG...", author="System").send()

        vector_manager = VectorStoreManager()
        index = vector_manager.load_index()

        generator = KudoResponseGenerator(index=index)

        cl.user_session.set("vector_manager", vector_manager)
        cl.user_session.set("generator", generator)

        stats = vector_manager.get_stats()

        await cl.Message(
            content=f"""✅ **Système prêt !**

📊 **Statistiques :**
- Documents indexés : {stats.get('total_documents', 0)} chunks
- Collection : {stats.get('collection_name')}
- Modèle LLM : {settings.llm_model}
- Embeddings : {settings.embedding_model}
""",
            author="System"
        ).send()

    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        await cl.Message(
            content=f"❌ **Erreur lors du chargement du système :** {e}\n\nVérifiez que l'index est créé avec `python scripts/pipeline.py index`",
            author="System"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Traitement des messages utilisateur avec streaming."""
    generator = cl.user_session.get("generator")

    if generator is None:
        await cl.Message(
            content="❌ Le système RAG n'est pas initialisé. Veuillez redémarrer l'application.",
            author="Assistant"
        ).send()
        return

    try:
        conversation_history = cl.user_session.get("history", [])

        msg = cl.Message(content="")
        await msg.send()

        stream = generator.generate_stream(
            question=message.content,
            include_sources=True,
            conversation_history=conversation_history,
        )

        nodes = []
        async for token, streamed_nodes in stream:
            if token:
                await msg.stream_token(token)
            nodes = streamed_nodes

        await msg.update()

        conversation_history.append({"role": "user", "content": message.content})
        conversation_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("history", conversation_history[-10:])

        confidence = generator._compute_confidence(nodes)
        num_sources = len(nodes)

        metadata_text = f"\n\n---\n📊 **Confiance:** {confidence:.1%} | 📚 **Sources:** {num_sources}"
        msg.content += metadata_text
        await msg.update()

        if nodes:
            source_elements = []

            for idx, node in enumerate(nodes, 1):
                metadata = node.node.metadata
                section = metadata.get("section", "N/A")
                category = metadata.get("category", "N/A")
                article_reference = metadata.get("article_reference", "N/A")
                excerpt = node.node.get_content()[:400]
                score = node.score

                source_content = f"""**Section:** {section}
**Catégorie:** {category}
**Référence:** {article_reference}
**Score de pertinence:** {score:.3f}

---

**Extrait du règlement:**

{excerpt}
"""

                source_elements.append(
                    cl.Text(
                        content=source_content,
                        name=f"📄 Source {idx}: {section}",
                        display="side"
                    )
                )

            await cl.ElementSidebar.set_title("📚 Sources consultées")
            await cl.ElementSidebar.set_elements(source_elements)

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")
        await cl.Message(
            content=f"❌ **Erreur:** {e}\n\nVeuillez réessayer ou reformuler votre question.",
            author="Assistant"
        ).send()


@cl.on_chat_end
def end():
    """Nettoyage à la fin du chat."""
    logger.info("Session de chat terminée")


@cl.on_settings_update
async def setup_settings(settings_update):
    """Mise à jour des paramètres utilisateur."""
    logger.info(f"Paramètres mis à jour: {settings_update}")


@cl.action_callback("feedback_positive")
async def on_positive_feedback(action: cl.Action):
    """Callback pour feedback positif."""
    logger.info(f"👍 Feedback positif reçu pour le message: {action.value}")

    await cl.Message(
        content="✅ Merci pour votre retour positif ! Cela m'aide à m'améliorer.",
        author="System"
    ).send()


@cl.action_callback("feedback_negative")
async def on_negative_feedback(action: cl.Action):
    """Callback pour feedback négatif."""
    logger.info(f"👎 Feedback négatif reçu pour le message: {action.value}")

    await cl.Message(
        content="⚠️ Merci pour votre retour. Pourriez-vous reformuler votre question pour que je puisse mieux vous aider ?",
        author="System"
    ).send()


if __name__ == "__main__":
    pass
