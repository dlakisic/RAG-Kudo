# RAG-Kudo 🥋

Système RAG (Retrieval-Augmented Generation) pour la formation des arbitres en Kudo. Ce projet utilise Docling pour l'ingestion de documents, LlamaIndex pour l'orchestration RAG, et OpenAI/Anthropic pour la génération de réponses pédagogiques.

## 🎯 Objectif

Créer un assistant intelligent pour la formation des arbitres de Kudo qui :
- Répond aux questions sur les règles d'arbitrage
- Cite les sources officielles du règlement
- Fournit des exemples concrets et des explications pédagogiques
- Génère des quiz d'entraînement
- Explique les décisions d'arbitrage

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│  Documents Sources (PDF, DOCX, MD)         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Docling (Extraction structurée)           │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Semantic Chunking (LlamaIndex)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  ChromaDB (Base vectorielle)               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Retriever (Recherche hybride)             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  LLM (GPT-4/Claude) + Prompts pédagogiques │
└─────────────────────────────────────────────┘
```

## 📁 Structure du projet

```
RAG-Kudo/
├── data/
│   ├── raw/              # Documents sources (PDF, DOCX, etc.)
│   ├── processed/        # Documents traités par Docling
│   └── vectorstore/      # Base de données vectorielle ChromaDB
├── src/
│   ├── ingestion/        # Modules d'ingestion (Docling, chunking)
│   ├── retrieval/        # Vector store et retriever
│   ├── generation/       # LLM et générateur de réponses
│   ├── evaluation/       # Évaluation de la qualité RAG
│   └── utils/            # Utilitaires
├── config/
│   └── settings.py       # Configuration centralisée
├── scripts/
│   ├── demo.py           # Script de démonstration
│   └── pipeline.py       # Pipeline CLI principal
├── notebooks/            # Jupyter notebooks pour expérimentation
├── app/                  # Interface utilisateur (API/Streamlit)
├── tests/                # Tests unitaires
├── .env.example          # Template de configuration
└── pyproject.toml        # Configuration uv et dépendances
```

## 🚀 Installation

### Prérequis

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (gestionnaire de packages)
- Clé API OpenAI ou Anthropic

### Installation avec uv

```bash
# Cloner le repository
git clone <repo-url>
cd RAG-Kudo

# Créer l'environnement virtuel et installer les dépendances
uv venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate
uv sync

# Installer les dépendances de développement
uv sync --extra dev
```

### Configuration

1. Copier le fichier de configuration d'exemple :
```bash
cp .env.example .env
```

2. Éditer `.env` et ajouter vos clés API :
```bash
OPENAI_API_KEY=your-openai-api-key-here
# ou
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

3. Ajuster les autres paramètres selon vos besoins (voir `.env.example`).

## 📚 Utilisation

### 1. Ajouter des documents

Placez vos documents de règlement Kudo dans `data/raw/` :
```bash
cp /path/to/reglement_kudo.pdf data/raw/
```

Formats supportés : PDF, DOCX, Markdown, HTML

### 2. Pipeline complet

#### Option A : Script de démonstration interactif

```bash
python scripts/demo.py
```

Choisissez parmi :
1. Pipeline complet (ingestion + indexation + démo)
2. Indexation uniquement
3. Démo retrieval et génération
4. Générer un quiz

#### Option B : Pipeline CLI

```bash
# Pipeline complet en une commande
python scripts/pipeline.py full

# Ou étape par étape :

# 1. Ingérer les documents
python scripts/pipeline.py ingest

# 2. Créer l'index vectoriel
python scripts/pipeline.py index

# 3. Poser une question
python scripts/pipeline.py query "Quelles sont les techniques autorisées ?"

# 4. Mode interactif
python scripts/pipeline.py interactive

# 5. Afficher les statistiques
python scripts/pipeline.py stats
```

### 3. Utilisation programmatique

```python
from src.ingestion import DoclingProcessor, SemanticChunker
from src.retrieval import VectorStoreManager, KudoRetriever
from src.generation import KudoResponseGenerator
from config import settings

# 1. Ingestion
processor = DoclingProcessor(output_dir=settings.processed_data_dir)
documents = processor.process_directory(settings.raw_data_dir)

# 2. Chunking
chunker = SemanticChunker()
nodes = chunker.chunk_multiple_documents(documents)

# 3. Indexation
manager = VectorStoreManager()
index = manager.create_index(nodes)

# 4. Génération de réponse
generator = KudoResponseGenerator(index=index)
result = generator.generate("Quelles sont les règles de scoring ?")

print(result["answer"])
print(f"Confiance: {result['confidence']}")
print(f"Sources: {result['num_sources']}")
```

## 🎓 Fonctionnalités

### Réponses pédagogiques

Le système génère des réponses structurées incluant :
- La règle officielle exacte
- Le contexte et le raisonnement
- Des exemples concrets de situations
- Les erreurs courantes à éviter
- Les références précises du règlement

### Génération de quiz

```python
quiz = generator.generate_quiz_question(category="sanctions")
print(quiz["quiz"])
```

### Explication de décisions

```python
result = generator.explain_decision(
    situation="Un combattant frappe après l'arrêt de l'arbitre",
    decision="Avertissement donné"
)
print(result["explanation"])
```

### Recherche par catégorie

```python
retriever = KudoRetriever(index=index)
nodes = retriever.retrieve_by_category(
    query="sanctions possibles",
    category="sanctions"
)
```

## ⚙️ Configuration avancée

### Modèles LLM

Modifier dans `.env` :
```bash
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview

# Ou Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
```

### Embeddings

```bash
EMBEDDING_MODEL=text-embedding-3-small  # Recommandé
# ou text-embedding-3-large pour plus de précision
```

### Chunking sémantique

```bash
CHUNK_SIZE=800
CHUNK_OVERLAP=150
SEMANTIC_BUFFER_SIZE=1
SEMANTIC_BREAKPOINT_THRESHOLD=95
```

### Retrieval

```bash
TOP_K=5                      # Nombre de documents à récupérer
SIMILARITY_THRESHOLD=0.7     # Seuil de similarité minimum
USE_RERANKING=true          # Activer le re-ranking
```

## 📊 Évaluation

Le système inclut des métriques d'évaluation RAG :
- **Faithfulness** : Le LLM reste-t-il fidèle aux sources ?
- **Answer Relevancy** : La réponse répond-elle à la question ?
- **Context Precision** : Les chunks récupérés sont-ils pertinents ?
- **Context Recall** : Tous les chunks nécessaires sont-ils récupérés ?

```python
# TODO: Module d'évaluation à implémenter
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(generator)
metrics = evaluator.evaluate(test_questions)
```

## 🔧 Développement

### Installation des dépendances de développement

```bash
uv sync --extra dev
```

### Tests

```bash
pytest tests/
```

### Formatage du code

```bash
black src/ scripts/
ruff check src/ scripts/
```

### Notebooks Jupyter

```bash
jupyter notebook notebooks/
```

## 📝 Structure des métadonnées

Chaque chunk est enrichi avec :
- `source_file` : Fichier source
- `file_name` : Nom du fichier
- `section` : Section du document
- `category` : Catégorie (techniques_autorisees, sanctions, scoring, etc.)
- `article_reference` : Référence d'article (ex: "Article 5.3")
- `chunk_id` : Identifiant du chunk

Catégories détectées automatiquement :
- `techniques_autorisees`
- `sanctions`
- `scoring`
- `equipement`
- `regles_generales`

## 🚧 Roadmap

- [ ] Interface web Streamlit
- [ ] API REST FastAPI
- [ ] Module d'évaluation RAGAS
- [ ] Support multilingue
- [ ] Analyse de vidéos de combats
- [ ] Export de rapports PDF
- [ ] Mode quiz interactif avec tracking de progression

## 🤝 Contribution

Les contributions sont bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📄 Licence

[À définir]

## 📧 Contact

[À définir]

---

**Note** : Ce système est conçu pour la formation et l'assistance aux arbitres. Les décisions officielles doivent toujours être prises en référence directe au règlement officiel du Kudo.
