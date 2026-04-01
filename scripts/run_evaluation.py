"""
Script pour exécuter l'évaluation RAGAS sur le dataset de test.

Usage:
    # Toutes les métriques (par défaut)
    uv run python scripts/run_evaluation.py

    # Seulement faithfulness
    uv run python scripts/run_evaluation.py --metrics faithfulness

    # Plusieurs métriques
    uv run python scripts/run_evaluation.py --metrics faithfulness answer_relevancy

    # Lister les métriques disponibles
    uv run python scripts/run_evaluation.py --list-metrics

    # Afficher des résultats existants
    uv run python scripts/run_evaluation.py --results-file data/evaluation/results.csv
"""
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import pandas as pd
from src.retrieval import VectorStoreManager
from src.generation import KudoResponseGenerator
from src.evaluation import RagasEvaluator
from src.evaluation.ragas_evaluator import AVAILABLE_METRICS, METRICS_REQUIRING_GROUND_TRUTH
from config import settings


def load_test_dataset(dataset_path: str) -> tuple[list[str], list[str]]:
    """
    Charge le dataset de test depuis un fichier JSON.

    Args:
        dataset_path: Chemin vers le fichier JSON

    Returns:
        Tuple (questions, ground_truths)
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["questions"], data["ground_truths"]


def list_available_metrics():
    """Affiche les métriques disponibles."""
    logger.info("\n" + "=" * 60)
    logger.info("MÉTRIQUES RAGAS DISPONIBLES")
    logger.info("=" * 60)

    descriptions = {
        "faithfulness": "La réponse est-elle fidèle aux documents sources ?",
        "answer_relevancy": "La réponse répond-elle bien à la question ?",
        "context_precision": "Les documents récupérés sont-ils pertinents ?",
        "context_recall": "Tous les éléments nécessaires ont-ils été récupérés ?",
    }

    for name in AVAILABLE_METRICS.keys():
        requires_gt = "  [requiert ground_truth]" if name in METRICS_REQUIRING_GROUND_TRUTH else ""
        logger.info(f"  • {name}{requires_gt}")
        logger.info(f"    {descriptions.get(name, '')}")

    logger.info("\n" + "=" * 60)
    logger.info("EXEMPLES D'UTILISATION")
    logger.info("=" * 60)
    logger.info("  # Toutes les métriques")
    logger.info("  uv run python scripts/run_evaluation.py")
    logger.info("")
    logger.info("  # Seulement faithfulness (pas besoin de ground_truth)")
    logger.info("  uv run python scripts/run_evaluation.py --metrics faithfulness")
    logger.info("")
    logger.info("  # Faithfulness + Answer Relevancy")
    logger.info("  uv run python scripts/run_evaluation.py --metrics faithfulness answer_relevancy")
    logger.info("=" * 60 + "\n")


def display_results(results_df: pd.DataFrame, metrics: list[str] = None):
    """
    Affiche les résultats détaillés de l'évaluation.

    Args:
        results_df: DataFrame contenant les résultats RAGAS
        metrics: Liste des métriques à afficher (None = toutes)
    """
    if metrics is None:
        metrics = list(AVAILABLE_METRICS.keys())

    logger.info("\n" + "=" * 80)
    logger.info("RÉSULTATS DÉTAILLÉS PAR QUESTION")
    logger.info("=" * 80 + "\n")

    for idx, row in results_df.iterrows():
        question = row.get('user_input', row.get('question', 'N/A'))
        logger.info(f"Question {idx + 1}: {question[:80]}...")

        for metric in metrics:
            if metric in row:
                value = row[metric]
                if pd.notna(value):
                    logger.info(f"  {metric.replace('_', ' ').title():.<25} {value:.3f}")

        logger.info("")


def main():
    """Fonction principale d'évaluation."""
    parser = argparse.ArgumentParser(
        description="Évaluation RAGAS du système RAG-Kudo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Toutes les métriques
  uv run python scripts/run_evaluation.py

  # Seulement faithfulness (plus rapide, moins cher)
  uv run python scripts/run_evaluation.py --metrics faithfulness

  # Plusieurs métriques
  uv run python scripts/run_evaluation.py --metrics faithfulness answer_relevancy

  # Lister les métriques disponibles
  uv run python scripts/run_evaluation.py --list-metrics
        """
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(AVAILABLE_METRICS.keys()),
        help="Métriques à évaluer (par défaut: toutes)"
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="Affiche les métriques disponibles et quitte"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="Chemin vers un fichier CSV de résultats existant à afficher"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Chemin de sortie pour le fichier CSV de résultats"
    )

    args = parser.parse_args()

    # Mode liste des métriques
    if args.list_metrics:
        list_available_metrics()
        return

    # Mode affichage de résultats existants
    if args.results_file:
        logger.info("=" * 80)
        logger.info("AFFICHAGE DES RÉSULTATS RAGAS")
        logger.info("=" * 80)

        results_path = Path(args.results_file)
        if not results_path.exists():
            logger.error(f"Le fichier {results_path} n'existe pas!")
            return

        logger.info(f"Chargement des résultats: {results_path}")
        results_df = pd.read_csv(results_path)
        logger.info(f"✅ {len(results_df)} résultats chargés")

        # Déterminer les métriques à afficher
        metrics_to_show = args.metrics if args.metrics else list(AVAILABLE_METRICS.keys())

        # Afficher le résumé
        logger.info("\n" + "=" * 60)
        logger.info("RÉSUMÉ DE L'ÉVALUATION RAGAS")
        logger.info("=" * 60)

        for metric in metrics_to_show:
            if metric in results_df.columns:
                mean_score = results_df[metric].mean()
                logger.info(f"{metric.replace('_', ' ').title():.<40} {mean_score:.3f}")

        logger.info("=" * 60 + "\n")

        # Afficher les détails
        display_results(results_df, metrics_to_show)
        logger.success("\n✅ Affichage terminé!")
        return

    # Mode évaluation
    selected_metrics = args.metrics  # None = toutes les métriques

    logger.info("=" * 80)
    logger.info("ÉVALUATION RAGAS DU SYSTÈME RAG-KUDO")
    logger.info("=" * 80)

    if selected_metrics:
        logger.info(f"Métriques sélectionnées: {selected_metrics}")
    else:
        logger.info("Métriques: toutes (faithfulness, answer_relevancy, context_precision, context_recall)")

    dataset_path = Path(__file__).parent.parent / "data" / "evaluation" / "test_dataset.json"
    logger.info(f"Chargement du dataset: {dataset_path}")

    questions, ground_truths = load_test_dataset(str(dataset_path))
    logger.info(f"✅ Dataset chargé: {len(questions)} questions")

    # Vérifier si ground_truths est nécessaire
    needs_ground_truth = selected_metrics is None or bool(
        set(selected_metrics) & METRICS_REQUIRING_GROUND_TRUTH
    )

    logger.info("\nInitialisation du système RAG...")
    vector_manager = VectorStoreManager()
    index = vector_manager.load_index()
    generator = KudoResponseGenerator(index=index)
    logger.info("✅ Système RAG initialisé")

    logger.info("\nCréation de l'évaluateur RAGAS...")
    evaluator = RagasEvaluator(generator=generator, metrics=selected_metrics)
    logger.info("✅ Évaluateur créé")

    logger.info("\n" + "=" * 80)
    logger.info("DÉMARRAGE DE L'ÉVALUATION")
    logger.info("=" * 80 + "\n")

    results_df = evaluator.evaluate_dataset(
        test_questions=questions,
        ground_truths=ground_truths if needs_ground_truth else None,
    )

    # Déterminer le chemin de sortie
    if args.output:
        output_path = Path(args.output)
    else:
        # Ajouter le nom des métriques au fichier si sélection spécifique
        if selected_metrics and len(selected_metrics) < 4:
            suffix = "_" + "_".join(selected_metrics)
            output_path = Path(__file__).parent.parent / "data" / "evaluation" / f"results{suffix}.csv"
        else:
            output_path = Path(__file__).parent.parent / "data" / "evaluation" / "results.csv"

    # Éviter d'écraser un run précédent: ajouter un timestamp si le fichier existe déjà.
    if output_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        logger.warning(f"Fichier existant détecté, nouveau chemin de sortie: {output_path}")

    results_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Résultats sauvegardés dans: {output_path}")

    # Afficher les détails
    metrics_to_show = selected_metrics if selected_metrics else list(AVAILABLE_METRICS.keys())
    display_results(results_df, metrics_to_show)
    logger.success("\n✅ Évaluation terminée avec succès!")


if __name__ == "__main__":
    main()
