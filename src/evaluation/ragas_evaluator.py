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


class RagasEvaluator:
    """
    Évaluateur utilisant RAGAS pour mesurer les performances du RAG.
    
    Métriques évaluées:
    - Faithfulness: La réponse est-elle fidèle aux documents sources ?
    - Answer Relevancy: La réponse répond-elle bien à la question ?
    - Context Precision: Les documents récupérés sont-ils pertinents ?
    - Context Recall: Tous les éléments nécessaires ont-ils été récupérés ?
    """
    
    def __init__(
        self,
        generator: Optional[KudoResponseGenerator] = None,
    ):
        """
        Initialise l'évaluateur RAGAS.
        
        Args:
            generator: Générateur de réponses RAG
        """
        self.generator = generator
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        logger.info("RagasEvaluator initialisé avec 4 métriques")
    
    def evaluate_dataset(
        self,
        test_questions: List[str],
        ground_truths: List[str],
    ) -> pd.DataFrame:
        """
        Évalue le système RAG sur un dataset de test.
        
        Args:
            test_questions: Liste de questions de test
            ground_truths: Réponses de référence attendues
            
        Returns:
            DataFrame avec les résultats d'évaluation
        """
        if not self.generator:
            raise ValueError("Generator non initialisé")
        
        if len(test_questions) != len(ground_truths):
            raise ValueError("Le nombre de questions et de ground truths doit être identique")
        
        logger.info(f"Évaluation sur {len(test_questions)} questions")
        
        # Génération des réponses et récupération des contextes
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        for i, question in enumerate(test_questions):
            logger.info(f"[{i+1}/{len(test_questions)}] Traitement: {question[:50]}...")
            
            try:
                # Récupération des documents pertinents
                nodes = self.generator.retriever.retrieve(question)
                
                # Extraction du contexte
                contexts = [node.node.get_content() for node in nodes]
                
                # Génération de la réponse
                response = self.generator.generate(question)
                
                data["question"].append(question)
                data["answer"].append(response["answer"])
                data["contexts"].append(contexts)
                data["ground_truth"].append(ground_truths[i])
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la question {i+1}: {e}")
                continue
        
        # Création du dataset pour RAGAS
        dataset = Dataset.from_dict(data)
        
        logger.info("Lancement de l'évaluation RAGAS...")

        # S'assurer que OPENAI_API_KEY est disponible pour RAGAS
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
            logger.debug("OPENAI_API_KEY définie pour RAGAS")

        # Créer un LLM LlamaIndex compatible pour RAGAS
        try:
            # Créer un LLM LlamaIndex OpenAI
            llamaindex_llm = LlamaIndexOpenAI(
                model="gpt-4-turbo",
                temperature=0.0,
                api_key=settings.openai_api_key
            )

            # Wrapper LlamaIndex pour RAGAS
            ragas_llm = LlamaIndexLLMWrapper(llamaindex_llm)
            logger.debug("LLM RAGAS configuré avec LlamaIndex")

            # Évaluation avec RAGAS en passant le LLM
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=ragas_llm,
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation RAGAS: {e}")
            logger.exception(e)
            raise
        
        # Conversion en DataFrame
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
        
        # Calcul des moyennes pour chaque métrique
        for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
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
    
    # Chargement du système RAG
    vector_manager = VectorStoreManager()
    index = vector_manager.load_index()
    generator = KudoResponseGenerator(index=index)
    
    # Création de l'évaluateur
    evaluator = RagasEvaluator(generator=generator)
    
    # Questions de test
    test_questions = [
        "Quelles sont les techniques de frappe autorisées en Kudo ?",
        "Comment sont attribués les points lors d'un combat ?",
        "Quel est l'équipement obligatoire pour les combattants ?",
    ]
    
    # Ground truths (réponses attendues)
    ground_truths = [
        "Les techniques de frappe autorisées en Kudo incluent les coups de poing, coups de pied, coups de genou et coups de coude, portés avec contrôle sur les zones autorisées.",
        "Les points sont attribués selon l'efficacité des techniques: ippon (3 points) pour une technique décisive, waza-ari (2 points) pour une technique efficace, et yuko (1 point) pour une technique correcte.",
        "L'équipement obligatoire comprend le casque intégral (Super Safe), les gants de Kudo, la coquille, le protège-dents et le kimono réglementaire.",
    ]
    
    # Évaluation
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
