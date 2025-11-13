"""
Gestionnaire d'observabilité avec LangFuse.
Utilise LlamaIndex callback handler pour tracer automatiquement.
"""

from typing import Optional
from loguru import logger

from config import settings

try:
    from langfuse import Langfuse
    from llama_index.core import Settings as LlamaSettings
    from llama_index.core.callbacks import CallbackManager
    from llama_index.callbacks.langfuse import langfuse_callback_handler
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    LANGFUSE_AVAILABLE = False
    logger.warning(f"LangFuse non installé: {e}. Observabilité désactivée.")


class LangFuseManager:
    """
    Gestionnaire d'observabilité avec LangFuse utilisant OpenInference.

    Instrumente automatiquement toutes les opérations LlamaIndex :
    - Requêtes utilisateur
    - Reformulations de requêtes
    - Recherches vectorielles
    - Re-ranking
    - Appels LLM
    - Réponses générées
    """

    def __init__(self):
        """Initialise le gestionnaire LangFuse."""
        self.enabled = settings.langfuse_enabled and LANGFUSE_AVAILABLE
        self.client: Optional[Langfuse] = None
        self.callback_handler = None

        if not LANGFUSE_AVAILABLE:
            logger.info("LangFuse non disponible (packages non installés)")
            return

        if not self.enabled:
            logger.info("LangFuse désactivé dans la configuration")
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning(
                "Clés LangFuse non configurées. "
                "Définissez LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans .env"
            )
            self.enabled = False
            return

        self._initialize_langfuse()

    def _initialize_langfuse(self):
        """Initialise LangFuse avec callback handler."""
        try:
            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )

            auth_check = self.client.auth_check()
            if not auth_check:
                logger.error("Échec de l'authentification LangFuse")
                self.enabled = False
                return

            self.callback_handler = langfuse_callback_handler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )

            LlamaSettings.callback_manager = CallbackManager([self.callback_handler])

            logger.success(
                f"✅ LangFuse initialisé avec succès (callback handler) - Host: {settings.langfuse_host}"
            )

        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de LangFuse: {e}")
            logger.exception(e)
            self.enabled = False
            self.client = None
            self.callback_handler = None

    def is_enabled(self) -> bool:
        """Vérifie si LangFuse est activé et fonctionnel."""
        return self.enabled and self.client is not None

    def flush(self):
        """Force l'envoi des données à LangFuse."""
        if self.is_enabled() and self.client:
            try:
                self.client.flush()
                logger.debug("LangFuse flush effectué")
            except Exception as e:
                logger.error(f"Erreur lors du flush LangFuse: {e}")

    def shutdown(self):
        """Arrête proprement le callback handler."""
        if self.is_enabled():
            try:
                if self.callback_handler:
                    self.callback_handler.flush()
                if self.client:
                    self.client.flush()
                logger.info("LangFuse arrêté proprement")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt de LangFuse: {e}")


# Instance globale
_langfuse_manager: Optional[LangFuseManager] = None


def get_langfuse_manager() -> LangFuseManager:
    """
    Retourne l'instance globale du gestionnaire LangFuse.
    Crée l'instance si elle n'existe pas encore.
    """
    global _langfuse_manager
    if _langfuse_manager is None:
        _langfuse_manager = LangFuseManager()
    return _langfuse_manager


def reset_langfuse_manager():
    """Réinitialise le gestionnaire LangFuse (utile pour les tests)."""
    global _langfuse_manager
    if _langfuse_manager and _langfuse_manager.is_enabled():
        _langfuse_manager.shutdown()
    _langfuse_manager = None
