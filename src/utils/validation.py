"""
Utilitaires de validation pour le système RAG-Kudo.
"""

import sys
from loguru import logger
from config import settings
from src.utils.embeddings import requires_openai_embeddings


def validate_api_keys() -> bool:
    """
    Valide les clés API nécessaires selon la configuration.

    Returns:
        True si toutes les clés nécessaires sont configurées, False sinon
    """
    need_openai = settings.llm_provider == "openai" or requires_openai_embeddings()
    need_anthropic = settings.llm_provider == "anthropic"

    if need_openai and not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY non configurée alors qu'elle est requise par la configuration.\n"
            "Créez un fichier .env basé sur .env.example et ajoutez votre clé API."
        )
        return False

    if need_anthropic and not settings.anthropic_api_key:
        logger.error(
            "ANTHROPIC_API_KEY non configurée alors que llm_provider=anthropic."
        )
        return False

    return True


def require_api_keys():
    """
    Valide les clés API requises et quitte si non.
    """
    if not validate_api_keys():
        sys.exit(1)


def validate_openai_api_key() -> bool:
    """
    Compatibilité backward: alias vers validate_api_keys.
    """
    return validate_api_keys()


def require_openai_api_key():
    """
    Compatibilité backward: alias vers require_api_keys.
    """
    require_api_keys()
