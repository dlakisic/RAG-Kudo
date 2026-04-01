"""
Évaluateur RAGAS pour mesurer la qualité du système RAG.
"""

from typing import List, Dict, Optional
from loguru import logger
import pandas as pd
import os

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LlamaIndexLLMWrapper
from datasets import Dataset
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI

from config import settings
from src.generation import KudoResponseGenerator


# Mapping des noms de métriques vers les objets RAGAS
AVAILABLE_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

# Métriques qui nécessitent ground_truth
METRICS_REQUIRING_GROUND_TRUTH = {"context_precision", "context_recall"}


class RagasEvaluator:
    """
    Évaluateur utilisant RAGAS pour mesurer les performances du RAG.

    Métriques disponibles:
    - faithfulness: La réponse est-elle fidèle aux documents sources ?
    - answer_relevancy: La réponse répond-elle bien à la question ?
    - context_precision: Les documents récupérés sont-ils pertinents ?
    - context_recall: Tous les éléments nécessaires ont-ils été récupérés ?
    """

    def __init__(
        self,
        generator: Optional[KudoResponseGenerator] = None,
        metrics: Optional[List[str]] = None,
    ):
        """
        Initialise l'évaluateur RAGAS.

        Args:
            generator: Générateur de réponses RAG
            metrics: Liste des noms de métriques à utiliser.
                     Options: "faithfulness", "answer_relevancy",
                              "context_precision", "context_recall"
                     Si None, utilise toutes les métriques.
        """
        self.generator = generator

        if metrics is None:
            self.metric_names = list(AVAILABLE_METRICS.keys())
        else:
            invalid = set(metrics) - set(AVAILABLE_METRICS.keys())
            if invalid:
                raise ValueError(
                    f"Métriques invalides: {invalid}. "
                    f"Options: {list(AVAILABLE_METRICS.keys())}"
                )
            self.metric_names = metrics

        self.metrics = [AVAILABLE_METRICS[name] for name in self.metric_names]
        self.requires_ground_truth = bool(
            set(self.metric_names) & METRICS_REQUIRING_GROUND_TRUTH
        )

        logger.info(f"RagasEvaluator initialisé avec {len(self.metrics)} métrique(s): {self.metric_names}")
    
    def evaluate_dataset(
        self,
        test_questions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Évalue le système RAG sur un dataset de test.

        Args:
            test_questions: Liste de questions de test
            ground_truths: Réponses de référence attendues.
                           Requis uniquement pour context_precision et context_recall.

        Returns:
            DataFrame avec les résultats d'évaluation
        """
        if not self.generator:
            raise ValueError("Generator non initialisé")

        if self.requires_ground_truth and ground_truths is None:
            raise ValueError(
                f"ground_truths requis pour les métriques: "
                f"{set(self.metric_names) & METRICS_REQUIRING_GROUND_TRUTH}"
            )

        if ground_truths and len(test_questions) != len(ground_truths):
            raise ValueError("Le nombre de questions et de ground truths doit être identique")

        logger.info(f"Évaluation sur {len(test_questions)} questions")
        logger.info(f"Métriques: {self.metric_names}")

        data = {
            "question": [],
            "answer": [],
            "contexts": [],
        }

        if ground_truths:
            data["ground_truth"] = []

        for i, question in enumerate(test_questions):
            logger.info(f"[{i+1}/{len(test_questions)}] Traitement: {question[:50]}...")

            try:
                nodes = self.generator.retriever.retrieve(question)

                contexts = [node.node.get_content() for node in nodes]

                response = self.generator.generate(question, retrieved_nodes=nodes)

                data["question"].append(question)
                data["answer"].append(response["answer"])
                data["contexts"].append(contexts)

                if ground_truths:
                    data["ground_truth"].append(ground_truths[i])

            except Exception as e:
                logger.error(f"Erreur lors du traitement de la question {i+1}: {e}")
                continue

        dataset = Dataset.from_dict(data)

        logger.info("Lancement de l'évaluation RAGAS...")

        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            logger.debug("OPENAI_API_KEY définie pour RAGAS")

        try:
            llamaindex_llm = LlamaIndexOpenAI(
                model="gpt-4-turbo",
                temperature=0.0,
                api_key=settings.openai_api_key
            )

            ragas_llm = LlamaIndexLLMWrapper(llamaindex_llm)
            logger.debug("LLM RAGAS configuré avec LlamaIndex")

            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=ragas_llm,
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation RAGAS: {e}")
            logger.exception(e)
            raise

        results_df = results.to_pandas()
        
        logger.success("Évaluation terminée!")
        self._log_summary(results_df)
        
        return results_df
    
    def _log_summary(self, results_df: pd.DataFrame):
        """
        Affiche un résumé des résultats d'évaluation.

        Args:
            results_df: DataFrame avec les résultats
        """
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ DE L'ÉVALUATION RAGAS")
        logger.info("="*60)

        for metric in self.metric_names:
            if metric in results_df.columns:
                mean_score = results_df[metric].mean()
                logger.info(f"{metric.replace('_', ' ').title():.<40} {mean_score:.3f}")

        logger.info("="*60 + "\n")
    
    def evaluate_single(
        self,
        question: str,
        ground_truth: str,
    ) -> Dict[str, float]:
        """
        Évalue une seule question.
        
        Args:
            question: Question à évaluer
            ground_truth: Réponse de référence
            
        Returns:
            Dictionnaire avec les scores
        """
        results_df = self.evaluate_dataset(
            test_questions=[question],
            ground_truths=[ground_truth],
        )
        
        return results_df.iloc[0].to_dict()


def main():
    """Fonction de test du module."""
    from src.retrieval import VectorStoreManager
    from src.generation import KudoResponseGenerator

    vector_manager = VectorStoreManager()
    index = vector_manager.load_index()
    generator = KudoResponseGenerator(index=index)

    evaluator = RagasEvaluator(generator=generator)

    test_questions = [
        "Quelles sont les techniques de frappe autorisées en Kudo ?",
        "Comment sont attribués les points lors d'un combat ?",
        "Quel est l'équipement obligatoire pour les combattants ?",
    ]

    ground_truths = [
        "Les techniques de frappe autorisées en Kudo incluent les coups de poing, coups de pied, coups de genou et coups de coude, portés avec contrôle sur les zones autorisées.",
        "Les points sont attribués selon l'efficacité des techniques: ippon (3 points) pour une technique décisive, waza-ari (2 points) pour une technique efficace, et yuko (1 point) pour une technique correcte.",
        "L'équipement obligatoire comprend le casque intégral (Super Safe), les gants de Kudo, la coquille, le protège-dents et le kimono réglementaire.",
    ]

    results = evaluator.evaluate_dataset(
        test_questions=test_questions,
        ground_truths=ground_truths,
    )
    
    print("\n" + "="*80)
    print("RÉSULTATS DÉTAILLÉS")
    print("="*80)
    print(results)


if __name__ == "__main__":
    main()
