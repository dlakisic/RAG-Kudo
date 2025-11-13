"""
Interface Chainlit pour le système RAG-Kudo.
Chat interactif avec affichage des sources et feedback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
from loguru import logger
from llama_index.core.llms import ChatMessage, MessageRole

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

        nodes = generator.retriever.retrieve(message.content)

        context_str = "\n\n".join([
            f"[Source {i+1}] {node.node.get_content()}"
            for i, node in enumerate(nodes)
        ])

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="""Tu es un formateur expert en arbitrage de Kudo.

RÈGLES STRICTES À RESPECTER:
1. Tu dois UNIQUEMENT utiliser les informations présentes dans le contexte fourni
2. Si une information n'est PAS dans le contexte, tu DOIS dire "Je n'ai pas cette information dans le règlement fourni"
3. NE JAMAIS inventer, extrapoler ou ajouter des informations de ta connaissance générale
4. Cite EXACTEMENT les passages du règlement sans les reformuler de manière substantielle
5. Si le contexte ne contient pas assez d'informations pour répondre complètement, indique-le clairement

Format de réponse:
- Commence par citer la règle exacte du contexte
- Explique ensuite de manière pédagogique EN RESTANT FIDÈLE au texte source
- Si tu donnes un exemple, assure-toi qu'il est directement basé sur le contexte fourni"""
            ),
        ]

        for msg_dict in conversation_history[-6:]:
            role = MessageRole.USER if msg_dict["role"] == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg_dict["content"]))

        user_prompt = f"""CONTEXTE DU RÈGLEMENT OFFICIEL:
{context_str}

---

QUESTION DE L'UTILISATEUR:
{message.content}

---

INSTRUCTIONS:
Réponds à la question en utilisant UNIQUEMENT les informations présentes dans le contexte ci-dessus.
Si l'information n'est pas dans le contexte, indique-le clairement.
Cite les règles exactes du règlement."""

        messages.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

        msg = cl.Message(content="")
        await msg.send()

        response_stream = generator.llm_manager.llm.stream_chat(messages)

        for chunk in response_stream:
            if chunk.delta:
                await msg.stream_token(chunk.delta)

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
                article_ref = metadata.get("article_ref", "N/A")
                excerpt = node.node.get_content()[:400]
                score = node.score

                source_content = f"""**Section:** {section}
**Catégorie:** {category}
**Référence:** {article_ref}
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
