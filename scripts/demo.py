#!/usr/bin/env python3
"""
Script de démonstration du système RAG-Kudo.
Montre le pipeline complet : ingestion -> indexation -> retrieval -> génération.
"""

import sys
from pathlib import Path

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ingestion import DoclingProcessor, SemanticChunker
from src.retrieval import VectorStoreManager, KudoRetriever
from src.generation import LLMManager, KudoResponseGenerator
from config import settings


def demo_ingestion():
    """Démo du module d'ingestion."""
    logger.info("=== DEMO: Ingestion de documents ===\n")

    # Vérification de la présence de documents
    if not any(settings.raw_data_dir.iterdir()):
        logger.warning(
            f"Aucun document trouvé dans {settings.raw_data_dir}\n"
            "Ajoutez des PDFs ou documents dans data/raw/ pour continuer."
        )
        return []

    # Traitement avec Docling
    processor = DoclingProcessor(
        output_dir=settings.processed_data_dir,
        extract_tables=settings.docling_extract_tables,
        ocr_enabled=settings.docling_ocr_enabled,
    )

    documents = processor.process_directory(settings.raw_data_dir)
    logger.success(f"✓ {len(documents)} documents traités\n")

    # Découpage sémantique
    chunker = SemanticChunker()
    nodes = chunker.chunk_multiple_documents(documents)
    logger.success(f"✓ {len(nodes)} chunks créés\n")

    return nodes


def demo_indexation(nodes):
    """Démo de l'indexation vectorielle."""
    logger.info("=== DEMO: Indexation vectorielle ===\n")

    if not nodes:
        logger.warning("Aucun node à indexer. Chargement d'un index existant...\n")
        manager = VectorStoreManager()
        try:
            index = manager.load_index()
            logger.success("✓ Index existant chargé\n")
            return index
        except Exception:
            logger.error("Aucun index trouvé. Veuillez d'abord ingérer des documents.\n")
            return None

    # Création de l'index
    manager = VectorStoreManager()
    index = manager.create_index(nodes)

    # Affichage des stats
    stats = manager.get_stats()
    logger.info("Statistiques de l'index:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    logger.info("")

    return index


def demo_retrieval(index):
    """Démo du retrieval."""
    logger.info("=== DEMO: Retrieval de documents ===\n")

    if not index:
        logger.error("Pas d'index disponible pour le retrieval\n")
        return

    retriever = KudoRetriever(index=index)

    # Questions de test
    test_queries = [
        "Quelles sont les techniques de frappe autorisées ?",
        "Comment sont attribués les points ?",
    ]

    for query in test_queries:
        logger.info(f"Question: {query}")
        nodes = retriever.retrieve(query)

        logger.info(f"Résultats: {len(nodes)} documents trouvés")
        for i, node in enumerate(nodes[:2], 1):  # Afficher top 2
            logger.info(f"  [{i}] Score: {node.score:.3f}")
            logger.info(f"      Catégorie: {node.node.metadata.get('category', 'N/A')}")
            logger.info(f"      Texte: {node.node.get_content()[:100]}...")
        logger.info("")


def demo_generation(index):
    """Démo de la génération de réponses."""
    logger.info("=== DEMO: Génération de réponses ===\n")

    if not index:
        logger.error("Pas d'index disponible pour la génération\n")
        return

    generator = KudoResponseGenerator(index=index)

    # Question de test
    question = "Explique-moi les règles de scoring en Kudo"

    logger.info(f"Question: {question}\n")

    try:
        result = generator.generate(question, include_sources=True)

        logger.info("Réponse générée:")
        print(f"\n{'-'*80}")
        print(result["answer"])
        print(f"{'-'*80}\n")

        logger.info(f"Confiance: {result['confidence']:.2f}")
        logger.info(f"Sources utilisées: {result['num_sources']}")

        if result["sources"]:
            logger.info("\nSources principales:")
            for source in result["sources"][:2]:
                logger.info(f"  - {source['section']} (score: {source['relevance_score']})")

    except Exception as e:
        logger.error(f"Erreur lors de la génération: {e}")


def demo_quiz():
    """Démo de génération de quiz."""
    logger.info("\n=== DEMO: Génération de quiz ===\n")

    manager = VectorStoreManager()
    try:
        index = manager.load_index()
        generator = KudoResponseGenerator(index=index)

        quiz = generator.generate_quiz_question()

        print(f"\n{'-'*80}")
        print(quiz["quiz"])
        print(f"{'-'*80}\n")

        logger.info(f"Catégorie: {quiz.get('category', 'N/A')}")
        logger.info(f"Section: {quiz.get('source_section', 'N/A')}")

    except Exception as e:
        logger.error(f"Erreur: {e}")


def main():
    """Fonction principale."""
    logger.info("""
╔═══════════════════════════════════════════════════════════╗
║         RAG-KUDO - Système de Formation d'Arbitres        ║
║                     Démonstration                         ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Vérification de la configuration
    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY non configurée!\n"
            "Créez un fichier .env basé sur .env.example et ajoutez votre clé API."
        )
        return

    logger.info(f"Configuration:")
    logger.info(f"  LLM: {settings.llm_provider}/{settings.llm_model}")
    logger.info(f"  Embeddings: {settings.embedding_model}")
    logger.info(f"  Vector Store: {settings.vectorstore_type}")
    logger.info(f"  Data dir: {settings.raw_data_dir}\n")

    # Choix du mode
    print("Que voulez-vous faire ?")
    print("1. Pipeline complet (ingestion + indexation + démo)")
    print("2. Indexation uniquement (si documents déjà traités)")
    print("3. Démo retrieval et génération (si index existe)")
    print("4. Générer un quiz")

    try:
        choice = input("\nVotre choix (1-4): ").strip()

        if choice == "1":
            # Pipeline complet
            nodes = demo_ingestion()
            index = demo_indexation(nodes)
            if index:
                demo_retrieval(index)
                demo_generation(index)

        elif choice == "2":
            # Indexation seule
            nodes = demo_ingestion()
            demo_indexation(nodes)

        elif choice == "3":
            # Démo avec index existant
            manager = VectorStoreManager()
            index = manager.load_index()
            demo_retrieval(index)
            demo_generation(index)

        elif choice == "4":
            # Quiz
            demo_quiz()

        else:
            logger.warning("Choix invalide")

        logger.success("\n✓ Démonstration terminée!")

    except KeyboardInterrupt:
        logger.info("\n\nDémonstration interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"\n\nErreur: {e}")
        logger.exception("Détails de l'erreur:")


if __name__ == "__main__":
    main()
