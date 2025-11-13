# 📊 Évaluation du Système RAG-Kudo

## Vue d'ensemble

Ce document présente l'évaluation quantitative du système RAG-Kudo en utilisant le framework **RAGAS** (Retrieval-Augmented Generation Assessment).

L'évaluation mesure 4 métriques clés pour garantir la qualité et la fiabilité des réponses dans un contexte réglementaire.

---

## 🎯 Métriques RAGAS

### 1. **Faithfulness** (Fidélité aux Sources)
- **Définition** : Mesure si la réponse est fidèle aux documents sources sans ajout d'informations externes
- **Importance** : Critique dans un contexte réglementaire où l'exactitude est primordiale
- **Calcul** : % d'affirmations dans la réponse qui peuvent être vérifiées dans le contexte

### 2. **Answer Relevancy** (Pertinence de la Réponse)
- **Définition** : Mesure si la réponse répond bien à la question posée
- **Importance** : Garantit que l'utilisateur obtient l'information recherchée
- **Calcul** : Similarité cosinus entre la question et une question générée depuis la réponse

### 3. **Context Precision** (Précision du Contexte)
- **Définition** : Mesure si les documents récupérés sont pertinents pour la question
- **Importance** : Valide la qualité du système de retrieval
- **Calcul** : % de documents récupérés qui sont réellement utiles

### 4. **Context Recall** (Rappel du Contexte)
- **Définition** : Mesure si tous les éléments nécessaires ont été récupérés
- **Importance** : Garantit qu'aucune information importante n'est manquée
- **Calcul** : % d'informations de la ground truth présentes dans le contexte récupéré

---

## 📈 Résultats - Baseline (v1.0)

### Dataset de Test
- **Nombre de questions** : 10
- **Source** : Règlement officiel d'arbitrage Kudo (30 pages, PDF français)
- **Thématiques** : Scoring, équipement, techniques autorisées, durées, pénalités

### Résultats Quantitatifs

| Métrique | Score | Interprétation | Statut |
|----------|-------|----------------|--------|
| **Faithfulness** | **55.6%** | Environ 44% de la réponse contient des informations non présentes dans les sources | ⚠️ **PROBLÈME** |
| **Answer Relevancy** | **86.3%** | Les réponses sont pertinentes et répondent bien aux questions | ✅ BON |
| **Context Precision** | **71.8%** | Environ 72% des documents récupérés sont pertinents | ✅ BON |
| **Context Recall** | **80.0%** | 80% des informations nécessaires sont récupérées | ✅ BON |

### 🔍 Analyse des Résultats

#### ✅ Points Forts
- **Retrieval efficace** : Context Precision (71.8%) et Context Recall (80.0%) montrent que le système récupère bien les bons documents
- **Pertinence élevée** : Answer Relevancy (86.3%) indique que les réponses sont alignées avec les questions

#### ⚠️ Point Faible Identifié
- **Faithfulness faible (55.6%)** : Le LLM ajoute ~44% d'informations non présentes dans les sources
  - **Cause** : Prompts trop permissifs permettant au LLM d'utiliser ses connaissances générales
  - **Risque** : Génération d'informations incorrectes ou non vérifiées dans un contexte réglementaire

---

## 🔧 Optimisations Appliquées (v2.0)

### 1. **Renforcement des Prompts Système**

#### Avant
```python
SYSTEM_PROMPT = """Tu es un formateur expert en arbitrage de Kudo.
Réponds de manière claire et pédagogique en te basant sur le contexte fourni.
Structure ta réponse avec des exemples concrets."""
```

#### Après
```python
SYSTEM_PROMPT = """Tu es un formateur expert en arbitrage de Kudo.

RÈGLES STRICTES À RESPECTER:
1. Tu dois UNIQUEMENT utiliser les informations présentes dans le contexte fourni
2. Si une information n'est PAS dans le contexte, tu DOIS dire "Je n'ai pas cette information dans le règlement fourni"
3. NE JAMAIS inventer, extrapoler ou ajouter des informations de ta connaissance générale
4. Cite EXACTEMENT les passages du règlement sans les reformuler de manière substantielle
5. Si le contexte ne contient pas assez d'informations pour répondre complètement, indique-le clairement

Format de réponse:
- Commence par citer la règle exacte du contexte
- Explique ensuite de manière pédagogique EN RESTANT FIDÈLE au texte source
- Si tu donnes un exemple, assure-toi qu'il est directement basé sur le contexte fourni"""
```

### 2. **Réduction de la Température**
- **Avant** : `temperature = 0.1`
- **Après** : `temperature = 0.0`
- **Impact** : Réponses plus déterministes et moins créatives (moins d'hallucinations)

### 3. **Amélioration du Prompt Utilisateur**
- Séparation claire : `CONTEXTE` / `QUESTION` / `INSTRUCTIONS`
- Emphase sur "UNIQUEMENT les informations du contexte"
- Instruction explicite de signaler les informations manquantes

---

## 🎯 Résultats Attendus (v2.0)

### Hypothèses d'Amélioration

| Métrique | Baseline (v1.0) | Objectif (v2.0) | Amélioration Attendue |
|----------|-----------------|-----------------|----------------------|
| **Faithfulness** | 55.6% | **> 75%** | **+20 points** |
| Answer Relevancy | 86.3% | ~85% | Stable |
| Context Precision | 71.8% | ~72% | Stable |
| Context Recall | 80.0% | ~80% | Stable |

**Note** : Une légère baisse de Answer Relevancy est acceptable si elle résulte d'une plus grande prudence (réponses "je ne sais pas" quand l'info manque).

---

## 📝 Exemples de Questions du Dataset

### Question 1 - Scoring
**Question** : "Quelle est la valeur en points d'un ippon en Kudo ?"

**Ground Truth** : "Un ippon vaut 8 points en Kudo. Il est attribué uniquement en cas de soumission ou de KO/TKO."

**Évaluation** : Teste la capacité à extraire une information factuelle précise.

---

### Question 2 - Techniques Autorisées
**Question** : "Quelles sont les techniques de frappe autorisées en Kudo ?"

**Ground Truth** : "Les techniques de frappe autorisées en Kudo incluent les coups de poing, les coups de pied, les coups de genou, les coups de coude et les coups de tête."

**Évaluation** : Teste l'exhaustivité et la précision des listes.

---

### Question 3 - Règles Spécifiques
**Question** : "Dans quel cas les frappes génitales sont-elles autorisées ?"

**Ground Truth** : "Les frappes génitales sont interdites et constituent une faute, sauf dans un cas particulier spécifique : lorsqu'il y a un écart de 25 kg entre les deux adversaires et que cela est spécifié dès le début du combat."

**Évaluation** : Teste la capacité à gérer des cas exceptionnels et des nuances réglementaires.

---

## 🔄 Méthodologie d'Évaluation

### Pipeline d'Évaluation

```
1. Chargement du dataset (10 questions + ground truths)
   ↓
2. Pour chaque question:
   - Retrieval de documents pertinents
   - Extraction du contexte
   - Génération de la réponse
   ↓
3. Création du dataset RAGAS
   - question, answer, contexts, ground_truth
   ↓
4. Calcul des métriques RAGAS
   - Faithfulness, Answer Relevancy, Context Precision, Context Recall
   ↓
5. Analyse et sauvegarde des résultats
```

### Configuration
- **LLM pour génération** : `gpt-4-turbo` (température variable)
- **LLM pour évaluation RAGAS** : `gpt-4-turbo` (température 0.0)
- **Embeddings** : `text-embedding-3-small`
- **Re-ranker** : `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Top-k retrieval** : 5 documents

---

## 🛠️ Reproduction de l'Évaluation

### Prérequis
```bash
# API Keys nécessaires
export OPENAI_API_KEY="sk-..."

# Installation
uv sync
```

### Exécution
```bash
# Évaluation complète (10 questions)
uv run python scripts/run_evaluation.py

# Analyse des résultats
uv run python scripts/analyze_results.py

# Résultats sauvegardés dans
data/evaluation/results.csv
```

### Coûts Estimés
- **Baseline** : ~$0.50 - $1.00 (10 questions × 4 métriques)
- **Note** : RAGAS appelle GPT-4 plusieurs fois par question pour calculer les métriques

---

## 📚 Références

- [RAGAS Framework](https://docs.ragas.io/)
- [RAGAS Paper (arXiv)](https://arxiv.org/abs/2309.15217)
- [LlamaIndex RAGAS Integration](https://docs.llamaindex.ai/en/stable/examples/evaluation/ragas_evaluation/)

---

## 🔮 Prochaines Étapes

1. **Réévaluation post-optimisation** (nécessite recharge quota OpenAI)
2. **Évaluation humaine** pour validation qualitative
3. **A/B testing** avec arbitres réels
4. **Expansion du dataset** à 50-100 questions
5. **Fine-tuning** potentiel si Faithfulness reste < 80%
