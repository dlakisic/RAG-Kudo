"""
Utilitaires de validation pour le système RAG-Kudo.
"""

import sys
from loguru import logger
from config import settings


def validate_openai_api_key() -> bool:
    """
    Valide que la clé API OpenAI est configurée.

    Returns:
        True si la clé est configurée, False sinon
    """
    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY non configurée!\n"
            "Créez un fichier .env basé sur .env.example et ajoutez votre clé API."
        )
        return False
    return True


def require_openai_api_key():
    """
    Valide que la clé API OpenAI est configurée et quitte si non.
    """
    if not validate_openai_api_key():
        sys.exit(1)
