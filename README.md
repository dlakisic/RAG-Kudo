# RAG-Kudo 🥋

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12+-green.svg)](https://www.llamaindex.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Système RAG (Retrieval-Augmented Generation) avancé pour la formation des arbitres en Kudo. Utilise **LlamaIndex**, **Docling**, **RAGAS**, et **LangFuse** pour fournir des réponses précises et traçables basées sur le règlement officiel.

---

## 🎯 Objectif

Créer un assistant intelligent pour la formation des arbitres de Kudo qui :
- 📖 Répond aux questions sur les règles d'arbitrage avec **fidélité aux sources**
- 📚 Cite les sources officielles du règlement
- 🎓 Fournit des explications pédagogiques et des exemples concrets
- 📊 Mesure la qualité des réponses avec **RAGAS**
- 🔍 Traçabilité complète via **LangFuse**

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Documents Sources                          │
│              (PDF - Règlement Kudo Officiel)                   │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                   Docling Processor                            │
│         • Extraction structurée (texte + tables)               │
│         • OCR pour documents scannés                           │
│         • Détection automatique de sections                    │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                  Semantic Chunking                             │
│         • Chunking intelligent avec LlamaIndex                 │
│         • Préservation du contexte (800 tokens, overlap 150)   │
│         • Enrichissement métadonnées (sections, catégories)    │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                   Vector Store (ChromaDB)                      │
│         • Embeddings: text-embedding-3-small                   │
│         • Stockage local persistant                            │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│              Advanced Retrieval Pipeline                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Query            │→ │ Semantic Search  │→ │ Re-ranking   │ │
│  │ Reformulation    │  │ (Top-K = 10)     │  │ CrossEncoder │ │
│  │ (LLM-based)      │  │                  │  │ (Top-5)      │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                  Response Generation                           │
│         • LLM: GPT-4 Turbo (temperature=0.0)                   │
│         • Prompts optimisés pour fidélité aux sources          │
│         • Streaming support (Chainlit UI)                      │
└───────────────────────────┬────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────┐
│                 Observability & Evaluation                     │
│  ┌─────────────────────┐       ┌─────────────────────┐        │
│  │   LangFuse Traces   │       │   RAGAS Metrics     │        │
│  │  • LLM calls        │       │  • Faithfulness     │        │
│  │  • Retrieval logs   │       │  • Relevancy        │        │
│  │  • Latency tracking │       │  • Precision/Recall │        │
│  └─────────────────────┘       └─────────────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

---

## ✨ Caractéristiques Clés

### 🔬 Évaluation Quantitative (RAGAS)
- **Faithfulness**: 55.6% → Optimisation en cours (objectif: >75%)
- **Answer Relevancy**: 86.3%
- **Context Precision**: 71.8%
- **Context Recall**: 80.0%

Voir [EVALUATION.md](EVALUATION.md) pour les détails complets.

### 🔍 Observabilité (LangFuse)
- Traçabilité complète des appels LLM
- Monitoring de la latence et des coûts
- Débogage facilité des chaînes RAG

### 🎯 Retrieval Avancé
- **Query Reformulation**: Génération de variations de requêtes avec LLM
- **Re-ranking**: CrossEncoder pour améliorer la précision
- **Hybrid Search**: Combinaison sémantique + métadonnées

### 💬 Interface Interactive (Chainlit)
- Chat en temps réel avec streaming
- Affichage des sources et scores de pertinence
- Support multilingue (FR/EN/RU)

---

## 📊 Résultats d'Évaluation

| Métrique | Score Baseline | Statut | Détails |
|----------|---------------|--------|---------|
| **Faithfulness** | 55.6% | ⚠️ En amélioration | LLM ajoutait 44% d'infos externes → Prompts optimisés |
| **Answer Relevancy** | 86.3% | ✅ Excellent | Réponses pertinentes aux questions |
| **Context Precision** | 71.8% | ✅ Bon | Retrieval efficace |
| **Context Recall** | 80.0% | ✅ Bon | Peu d'informations manquées |

**Actions prises pour améliorer Faithfulness:**
1. Renforcement des prompts système (interdiction stricte d'inventer)
2. Réduction température: 0.1 → 0.0
3. Instructions explicites de citer exactement les sources

📈 [Voir l'analyse complète](EVALUATION.md)

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (gestionnaire de packages rapide)
- Clé API OpenAI (ou Anthropic)
- GPU recommandé pour re-ranking (optionnel)

### Installation Rapide

```bash
# Cloner le repository
git clone https://github.com/dlakisic/RAG-Kudo.git
cd RAG-Kudo

# Installer avec uv
uv sync

# Configuration
cp .env.example .env
# Éditer .env et ajouter votre OPENAI_API_KEY
```

### Configuration

Éditer `.env` :

```bash
# LLM
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.0

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small

# Retrieval
TOP_K=5
USE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# LangFuse (optionnel)
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
```

---

## 📚 Utilisation

### 1. Pipeline Complet (Ingestion → Indexation → Interface)

```bash
# Placer vos documents PDF dans data/raw/
cp /path/to/reglement_kudo.pdf data/raw/

# Lancer le pipeline
uv run python scripts/pipeline.py full

# Ou étape par étape:
uv run python scripts/pipeline.py ingest  # Extraction avec Docling
uv run python scripts/pipeline.py index   # Indexation vectorielle
uv run python scripts/pipeline.py query "Quelle est la valeur d'un ippon ?"
```

### 2. Interface Web (Chainlit)

```bash
# Lancer l'interface
chainlit run app/chainlit_app.py -w

# Accéder à http://localhost:8000
```

**Fonctionnalités de l'interface:**
- 💬 Chat en temps réel avec streaming
- 📚 Affichage des sources dans la sidebar
- 📊 Scores de confiance et de pertinence
- 🌍 Support FR/EN/RU

### 3. Évaluation RAGAS

```bash
# Évaluer le système sur 10 questions
uv run python scripts/run_evaluation.py

# Analyser les résultats
uv run python scripts/analyze_results.py

# Résultats dans: data/evaluation/results.csv
```

### 4. Utilisation Programmatique

```python
from src.retrieval import VectorStoreManager
from src.generation import KudoResponseGenerator

# Charger l'index
manager = VectorStoreManager()
index = manager.load_index()

# Générer une réponse
generator = KudoResponseGenerator(index=index)
result = generator.generate("Quelles sont les techniques de frappe autorisées ?")

print(result["answer"])
print(f"Confiance: {result['confidence']:.1%}")
print(f"Sources: {result['num_sources']}")
```

---

## 📁 Structure du Projet

```
RAG-Kudo/
├── data/
│   ├── raw/                 # Documents sources (PDF)
│   ├── processed/           # Documents traités (Docling)
│   ├── vectorstore/         # ChromaDB
│   └── evaluation/          # Résultats RAGAS
├── src/
│   ├── ingestion/           # Docling processor + chunking
│   ├── retrieval/           # Vector store + retriever + re-ranker
│   ├── generation/          # LLM manager + response generator
│   ├── evaluation/          # RAGAS evaluator
│   ├── observability/       # LangFuse integration
│   └── utils/               # Helpers (GPU utils, etc.)
├── app/
│   └── chainlit_app.py      # Interface web Chainlit
├── scripts/
│   ├── pipeline.py          # Pipeline CLI principal
│   ├── run_evaluation.py    # Évaluation RAGAS
│   └── analyze_results.py   # Analyse des résultats
├── config/
│   └── settings.py          # Configuration Pydantic
├── EVALUATION.md            # 📊 Rapport d'évaluation détaillé
├── FEATURES.md              # Liste des fonctionnalités
├── QUICKSTART.md            # Guide de démarrage rapide
└── README.md                # Ce fichier
```

---

## 🔧 Composants Techniques

### Ingestion (Docling)
- Extraction structurée de PDFs (texte + tables)
- OCR pour documents scannés
- Détection automatique de sections et métadonnées

### Retrieval Pipeline
1. **Query Reformulation**: LLM génère des variations de la question
2. **Semantic Search**: Embeddings + similarité cosinus (Top-10)
3. **Re-ranking**: CrossEncoder affine les résultats (Top-5)

### Generation
- **Prompts optimisés** pour fidélité aux sources
- **Temperature 0.0** pour réponses déterministes
- **Citations explicites** du règlement

### Observabilité
- **LangFuse**: Traces LLM, latence, coûts
- **RAGAS**: Évaluation quantitative (4 métriques)
- **Logs structurés** avec Loguru

---

## 📊 Métriques & Performances

### Latence (sur GPU T4)
- Ingestion: ~2-3s par page PDF
- Retrieval: ~300-500ms
- Generation: ~2-4s (streaming)
- **Total**: ~3-5s par requête

### Coûts (estimation)
- Embeddings: ~$0.0001 par chunk
- LLM (GPT-4 Turbo): ~$0.01-0.03 par requête
- RAGAS évaluation: ~$0.50-1.00 pour 10 questions

---

## 🎯 Cas d'Usage

### 1. Formation d'Arbitres
- Questions/réponses sur les règles
- Explications pédagogiques avec exemples
- Citations exactes du règlement officiel

### 2. Vérification de Décisions
```python
result = generator.explain_decision(
    situation="Combattant frappe après l'arrêt",
    decision="Avertissement donné"
)
```

### 3. Génération de Quiz
```python
quiz = generator.generate_quiz_question(category="scoring")
```

---

## 🔬 Challenges & Solutions

| Challenge | Solution Implémentée |
|-----------|---------------------|
| **Faithfulness faible (55.6%)** | Prompts stricts + température 0.0 + instructions explicites |
| **Contexte réglementaire** | Chunking sémantique + métadonnées enrichies |
| **Multilingue (FR/EN/RU)** | Prompts adaptatifs + support Chainlit |
| **Latence retrieval** | Re-ranking sur GPU + cache |
| **Traçabilité** | LangFuse pour observabilité complète |

---

## 🚧 Roadmap

- [x] Pipeline d'ingestion (Docling)
- [x] Retrieval avancé (reformulation + re-ranking)
- [x] Interface Chainlit
- [x] Évaluation RAGAS
- [x] Observabilité LangFuse
- [ ] Fine-tuning pour Faithfulness >80%
- [ ] API REST (FastAPI)
- [ ] Dataset étendu (50-100 questions)
- [ ] Support vidéo (analyse de combats)
- [ ] Déploiement cloud (Docker + K8s)

---

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

---

## 📄 Licence

MIT License - voir [LICENSE](LICENSE) pour détails.

---

## 📧 Contact

**Dino Lakisic** - [GitHub](https://github.com/dlakisic)

---

## 🙏 Remerciements

- [LlamaIndex](https://www.llamaindex.ai/) pour l'orchestration RAG
- [Docling](https://github.com/DS4SD/docling) pour l'extraction de documents
- [RAGAS](https://docs.ragas.io/) pour l'évaluation
- [LangFuse](https://langfuse.com/) pour l'observabilité
- Fédération de Kudo pour les documents officiels

---

**Note**: Ce système est conçu pour la formation et l'assistance. Les décisions officielles doivent toujours être prises en référence directe au règlement officiel du Kudo.
