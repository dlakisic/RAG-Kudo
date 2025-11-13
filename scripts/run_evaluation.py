"""
Script pour exécuter l'évaluation RAGAS sur le dataset de test.
"""
import sys
import json
from pathlib import Path

# Ajout du répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.retrieval import VectorStoreManager
from src.generation import KudoResponseGenerator
from src.evaluation import RagasEvaluator
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


def main():
    """Fonction principale d'évaluation."""
    logger.info("=" * 80)
    logger.info("ÉVALUATION RAGAS DU SYSTÈME RAG-KUDO")
    logger.info("=" * 80)

    # Chargement du dataset de test
    dataset_path = Path(__file__).parent.parent / "data" / "evaluation" / "test_dataset.json"
    logger.info(f"Chargement du dataset: {dataset_path}")

    questions, ground_truths = load_test_dataset(str(dataset_path))
    logger.info(f"✅ Dataset chargé: {len(questions)} questions")

    # Initialisation du système RAG
    logger.info("\nInitialisation du système RAG...")
    vector_manager = VectorStoreManager()
    index = vector_manager.load_index()
    generator = KudoResponseGenerator(index=index)
    logger.info("✅ Système RAG initialisé")

    # Création de l'évaluateur
    logger.info("\nCréation de l'évaluateur RAGAS...")
    evaluator = RagasEvaluator(generator=generator)
    logger.info("✅ Évaluateur créé")

    # Exécution de l'évaluation
    logger.info("\n" + "=" * 80)
    logger.info("DÉMARRAGE DE L'ÉVALUATION")
    logger.info("=" * 80 + "\n")

    results_df = evaluator.evaluate_dataset(
        test_questions=questions,
        ground_truths=ground_truths,
    )

    # Sauvegarde des résultats
    output_path = Path(__file__).parent.parent / "data" / "evaluation" / "results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n💾 Résultats sauvegardés dans: {output_path}")

    # Affichage des résultats détaillés
    logger.info("\n" + "=" * 80)
    logger.info("RÉSULTATS DÉTAILLÉS PAR QUESTION")
    logger.info("=" * 80 + "\n")

    for idx, row in results_df.iterrows():
        logger.info(f"Question {idx + 1}: {row['question'][:80]}...")
        logger.info(f"  Faithfulness:       {row.get('faithfulness', 'N/A'):.3f}")
        logger.info(f"  Answer Relevancy:   {row.get('answer_relevancy', 'N/A'):.3f}")
        logger.info(f"  Context Precision:  {row.get('context_precision', 'N/A'):.3f}")
        logger.info(f"  Context Recall:     {row.get('context_recall', 'N/A'):.3f}")
        logger.info("")

    logger.success("\n✅ Évaluation terminée avec succès!")


if __name__ == "__main__":
    main()
