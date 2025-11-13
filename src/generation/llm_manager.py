"""
Gestionnaire de modèles LLM.
Support pour OpenAI et Anthropic.
"""

from typing import Optional, Literal
from loguru import logger

from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.llms import LLM
from llama_index.core import Settings as LlamaSettings

from config import settings


class LLMManager:
    """Gestionnaire de modèles de langage."""

    def __init__(
        self,
        provider: Optional[Literal["openai", "anthropic"]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialise le gestionnaire LLM.

        Args:
            provider: Fournisseur LLM (openai ou anthropic)
            model: Nom du modèle
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens
        """
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.max_tokens

        self.llm = self._initialize_llm()

        logger.info(
            f"LLM initialisé: {self.provider}/{self.model} "
            f"(temp={self.temperature}, max_tokens={self.max_tokens})"
        )

    def _initialize_llm(self) -> LLM:
        """
        Initialise le modèle LLM selon le fournisseur.

        Returns:
            Instance du LLM
        """
        if self.provider == "openai":
            return OpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key,
                callback_manager=LlamaSettings.callback_manager,
            )
        elif self.provider == "anthropic":
            return Anthropic(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.anthropic_api_key,
                callback_manager=LlamaSettings.callback_manager,
            )
        else:
            raise ValueError(f"Provider non supporté: {self.provider}")

    def get_llm(self) -> LLM:
        """
        Retourne l'instance du LLM.

        Returns:
            Instance LLM
        """
        return self.llm

    def chat(self, messages: list) -> str:
        """
        Envoie des messages au LLM et retourne la réponse.

        Args:
            messages: Liste de messages (format ChatMessage)

        Returns:
            Réponse du LLM
        """
        try:
            response = self.llm.chat(messages)
            return response.message.content

        except Exception as e:
            logger.error(f"Erreur lors de l'appel LLM: {e}")
            raise

    def complete(self, prompt: str) -> str:
        """
        Génère une complétion à partir d'un prompt.

        Args:
            prompt: Prompt à compléter

        Returns:
            Complétion générée
        """
        try:
            response = self.llm.complete(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Erreur lors de la complétion: {e}")
            raise

    def stream_chat(self, messages: list):
        """
        Stream des réponses du LLM.

        Args:
            messages: Liste de messages

        Yields:
            Chunks de la réponse
        """
        try:
            response = self.llm.stream_chat(messages)
            for chunk in response:
                yield chunk.message.content

        except Exception as e:
            logger.error(f"Erreur lors du streaming: {e}")
            raise


def main():
    """Fonction de test du module."""
    from llama_index.core.llms import ChatMessage, MessageRole

    # Initialisation du LLM
    llm_manager = LLMManager()

    # Test de chat
    messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="Tu es un expert en arbitrage de Kudo.",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Explique-moi brièvement ce qu'est le Kudo.",
        ),
    ]

    print("Test du LLM:")
    print("-" * 60)
    response = llm_manager.chat(messages)
    print(response)
    print("-" * 60)

    # Test de streaming
    print("\nTest du streaming:")
    print("-" * 60)
    messages.append(
        ChatMessage(
            role=MessageRole.USER,
            content="Quelles sont les principales règles ?",
        )
    )

    for chunk in llm_manager.stream_chat(messages):
        print(chunk, end="", flush=True)
    print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
