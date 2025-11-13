#!/usr/bin/env python3
"""
Pipeline principal pour le système RAG-Kudo.
Commandes pour ingérer, indexer et interroger le système.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.ingestion import DoclingProcessor, SemanticChunker
from src.retrieval import VectorStoreManager, KudoRetriever
from src.generation import KudoResponseGenerator
from config import settings


def ingest_documents(input_dir: Path, recursive: bool = True):
    """
    Ingère et traite les documents.

    Args:
        input_dir: Répertoire contenant les documents
        recursive: Traiter les sous-répertoires
    """
    logger.info(f"Ingestion des documents depuis: {input_dir}")

    # Traitement avec Docling
    processor = DoclingProcessor(
        output_dir=settings.processed_data_dir,
        extract_tables=settings.docling_extract_tables,
        ocr_enabled=settings.docling_ocr_enabled,
    )

    documents = processor.process_directory(input_dir, recursive=recursive)
    logger.success(f"✓ {len(documents)} documents traités")

    return documents


def create_index(documents: list = None, force: bool = False):
    """
    Crée ou met à jour l'index vectoriel.

    Args:
        documents: Documents à indexer (si None, charge les documents traités)
        force: Forcer la recréation de l'index
    """
    logger.info("Création de l'index vectoriel")

    manager = VectorStoreManager()

    # Suppression de l'index existant si force=True
    if force:
        try:
            manager.delete_collection()
            logger.info("Index existant supprimé")
        except Exception:
            pass

    # Chargement des documents si non fournis
    if documents is None:
        import json
        processed_files = list(settings.processed_data_dir.glob("*_processed.json"))

        if not processed_files:
            logger.error("Aucun document traité trouvé. Exécutez d'abord l'ingestion.")
            return None

        documents = []
        for file_path in processed_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(json.load(f))

        logger.info(f"Chargé {len(documents)} documents traités")

    # Découpage en chunks
    chunker = SemanticChunker()
    nodes = chunker.chunk_multiple_documents(documents)
    logger.success(f"✓ {len(nodes)} chunks créés")

    # Création de l'index
    index = manager.create_index(nodes)

    # Affichage des stats
    stats = manager.get_stats()
    logger.info("Statistiques de l'index:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    return index


def query_system(question: str, show_sources: bool = True):
    """
    Interroge le système RAG.

    Args:
        question: Question à poser
        show_sources: Afficher les sources
    """
    logger.info(f"Question: {question}")

    # Chargement de l'index
    manager = VectorStoreManager()
    try:
        index = manager.load_index()
    except Exception as e:
        logger.error(f"Impossible de charger l'index: {e}")
        logger.info("Créez d'abord l'index avec: python scripts/pipeline.py index")
        return

    # Génération de la réponse
    generator = KudoResponseGenerator(index=index)
    result = generator.generate(question, include_sources=show_sources)

    # Affichage de la réponse
    print(f"\n{'='*80}")
    print(f"Question: {result['question']}")
    print(f"{'='*80}\n")
    print(result['answer'])
    print(f"\n{'='*80}\n")

    print(f"Confiance: {result['confidence']:.2%}")
    print(f"Sources utilisées: {result['num_sources']}")

    if show_sources and result['sources']:
        print("\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] {source['section']}")
            print(f"    Catégorie: {source['category']}")
            print(f"    Référence: {source['article_ref']}")
            print(f"    Score: {source['relevance_score']:.3f}")
            print(f"    Extrait: {source['excerpt'][:150]}...")


def interactive_mode():
    """Mode interactif pour poser plusieurs questions."""
    logger.info("Mode interactif - Tapez 'quit' ou 'exit' pour quitter")

    # Chargement de l'index
    manager = VectorStoreManager()
    try:
        index = manager.load_index()
    except Exception as e:
        logger.error(f"Impossible de charger l'index: {e}")
        return

    generator = KudoResponseGenerator(index=index)
    conversation_history = []

    print("\n" + "="*80)
    print("RAG-KUDO - Mode Interactif")
    print("="*80 + "\n")

    while True:
        try:
            question = input("\n🥋 Votre question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Au revoir!")
                break

            if not question:
                continue

            # Génération de la réponse
            result = generator.generate(
                question,
                include_sources=True,
                conversation_history=conversation_history,
            )

            # Affichage
            print(f"\n💡 Réponse:\n{result['answer']}\n")

            # Mise à jour de l'historique
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": result['answer']})

        except KeyboardInterrupt:
            print("\n")
            logger.info("Au revoir!")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")


def main():
    """Fonction principale avec CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline RAG-Kudo pour la formation d'arbitres",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Ingérer les documents depuis data/raw/
  python scripts/pipeline.py ingest

  # Créer l'index vectoriel
  python scripts/pipeline.py index

  # Créer l'index en forçant la recréation
  python scripts/pipeline.py index --force

  # Pipeline complet (ingestion + indexation)
  python scripts/pipeline.py full

  # Poser une question
  python scripts/pipeline.py query "Quelles sont les règles de scoring ?"

  # Mode interactif
  python scripts/pipeline.py interactive
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')

    # Commande: ingest
    ingest_parser = subparsers.add_parser('ingest', help='Ingérer les documents')
    ingest_parser.add_argument(
        '--input-dir',
        type=Path,
        default=settings.raw_data_dir,
        help='Répertoire des documents sources'
    )
    ingest_parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Ne pas traiter les sous-répertoires'
    )

    # Commande: index
    index_parser = subparsers.add_parser('index', help='Créer l\'index vectoriel')
    index_parser.add_argument(
        '--force',
        action='store_true',
        help='Forcer la recréation de l\'index'
    )

    # Commande: full
    subparsers.add_parser('full', help='Pipeline complet (ingest + index)')

    # Commande: query
    query_parser = subparsers.add_parser('query', help='Poser une question')
    query_parser.add_argument('question', type=str, help='Question à poser')
    query_parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Ne pas afficher les sources'
    )

    # Commande: interactive
    subparsers.add_parser('interactive', help='Mode interactif')

    # Commande: stats
    subparsers.add_parser('stats', help='Afficher les statistiques de l\'index')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Vérification de la configuration
    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY non configurée!\n"
            "Créez un fichier .env basé sur .env.example"
        )
        return

    # Exécution des commandes
    try:
        if args.command == 'ingest':
            ingest_documents(args.input_dir, recursive=not args.no_recursive)

        elif args.command == 'index':
            create_index(force=args.force)

        elif args.command == 'full':
            documents = ingest_documents(settings.raw_data_dir)
            create_index(documents=documents)

        elif args.command == 'query':
            query_system(args.question, show_sources=not args.no_sources)

        elif args.command == 'interactive':
            interactive_mode()

        elif args.command == 'stats':
            manager = VectorStoreManager()
            stats = manager.get_stats()
            print("\nStatistiques de l'index:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Erreur: {e}")
        logger.exception("Détails:")
        sys.exit(1)


if __name__ == "__main__":
    main()
