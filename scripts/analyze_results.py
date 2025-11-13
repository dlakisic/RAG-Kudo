"""
Script pour analyser les résultats de l'évaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json

results_path = Path(__file__).parent.parent / "data" / "evaluation" / "results.csv"
df = pd.read_csv(results_path)

print("=" * 80)
print("ANALYSE DES RÉSULTATS D'ÉVALUATION")
print("=" * 80)
print(f"\nNombre de questions évaluées: {len(df)}")
print(f"\nColonnes disponibles: {list(df.columns)}")

print("\n" + "=" * 80)
print("APERÇU DES RÉPONSES GÉNÉRÉES")
print("=" * 80)

for idx, row in df.iterrows():
    question = row['user_input']
    response = row['response'][:300] + "..." if len(row['response']) > 300 else row['response']
    ground_truth = row['reference'][:200] + "..." if len(row['reference']) > 200 else row['reference']

    print(f"\n{'='*80}")
    print(f"Question {idx + 1}: {question}")
    print(f"{'-'*80}")
    print(f"Réponse générée: {response}")
    print(f"{'-'*80}")
    print(f"Ground truth: {ground_truth}")
    print(f"{'-'*80}")

    if 'faithfulness' in df.columns:
        print(f"Métriques:")
        print(f"  - Faithfulness: {row.get('faithfulness', 'N/A')}")
        print(f"  - Answer Relevancy: {row.get('answer_relevancy', 'N/A')}")
        print(f"  - Context Precision: {row.get('context_precision', 'N/A')}")
        print(f"  - Context Recall: {row.get('context_recall', 'N/A')}")
