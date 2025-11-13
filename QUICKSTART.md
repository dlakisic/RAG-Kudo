# Guide de démarrage rapide 🚀

Ce guide vous permet de lancer le système RAG-Kudo en quelques minutes.

## Installation rapide

```bash
# 1. Cloner le repository
git clone <repo-url>
cd RAG-Kudo

# 2. Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Créer l'environnement et installer les dépendances
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# 4. Configurer les clés API
cp .env.example .env
# Éditer .env et ajouter votre OPENAI_API_KEY
```

## Premiers pas

### 1. Ajouter vos documents

Placez vos PDFs de règlement Kudo dans `data/raw/` :
```bash
cp /chemin/vers/reglement_kudo.pdf data/raw/
```

### 2. Lancer le pipeline complet

```bash
python scripts/pipeline.py full
```

Cette commande va :
1. Extraire le contenu des documents avec Docling
2. Découper intelligemment en chunks sémantiques
3. Créer l'index vectoriel dans ChromaDB
4. Vous permettre de poser des questions

### 3. Tester le système

#### Mode interactif (recommandé)

```bash
python scripts/pipeline.py interactive
```

Posez vos questions et obtenez des réponses avec sources :
```
🥋 Votre question: Quelles sont les techniques de frappe autorisées ?
💡 Réponse: [réponse détaillée avec exemples et références]
```

#### Question unique

```bash
python scripts/pipeline.py query "Comment sont attribués les points ?"
```

#### Script de démonstration

```bash
python scripts/demo.py
```

## Exemples de questions

Voici quelques questions types pour tester le système :

**Règles générales :**
- "Explique-moi les règles de base du Kudo"
- "Quelle est la durée d'un combat ?"
- "Combien d'arbitres sont nécessaires ?"

**Techniques :**
- "Quelles sont les techniques de frappe autorisées ?"
- "Les projections sont-elles autorisées ?"
- "Peut-on frapper au sol ?"

**Scoring :**
- "Comment marque-t-on des points en Kudo ?"
- "Qu'est-ce qu'un ippon ?"
- "Quelle est la différence entre waza-ari et ippon ?"

**Sanctions :**
- "Quelles sont les sanctions possibles ?"
- "Que se passe-t-il en cas de faute ?"
- "Quand donne-t-on un shido ?"

**Équipement :**
- "Quel équipement de protection est obligatoire ?"
- "La tenue est-elle réglementée ?"

## Fonctionnalités avancées

### Générer un quiz

```bash
# Dans le script demo.py, choisir l'option 4
python scripts/demo.py
```

### Utilisation programmatique

```python
from src.retrieval import VectorStoreManager
from src.generation import KudoResponseGenerator

# Charger l'index
manager = VectorStoreManager()
index = manager.load_index()

# Créer le générateur
generator = KudoResponseGenerator(index=index)

# Poser une question
result = generator.generate("Quelles sont les règles de scoring ?")
print(result["answer"])
```

### Recherche par catégorie

```python
from src.retrieval import KudoRetriever

retriever = KudoRetriever(index=index)
nodes = retriever.retrieve_by_category(
    query="règles de sanctions",
    category="sanctions"
)
```

## Commandes utiles

```bash
# Voir les statistiques de l'index
python scripts/pipeline.py stats

# Réingérer les documents (si vous ajoutez de nouveaux fichiers)
python scripts/pipeline.py ingest

# Recréer l'index depuis zéro
python scripts/pipeline.py index --force

# Pipeline complet
python scripts/pipeline.py full
```

## Personnalisation rapide

### Changer le modèle LLM

Éditer `.env` :
```bash
LLM_MODEL=gpt-4o              # Plus rapide
LLM_MODEL=gpt-4-turbo-preview # Plus puissant
```

### Utiliser Claude

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your-key-here
```

### Ajuster le nombre de sources

```bash
TOP_K=10  # Plus de contexte (par défaut: 5)
```

### Modifier la température

```bash
LLM_TEMPERATURE=0.0  # Plus déterministe
LLM_TEMPERATURE=0.3  # Plus créatif (par défaut: 0.1)
```

## Résolution de problèmes

### "No module named 'src'"

```bash
# Assurez-vous d'être dans le bon répertoire
cd RAG-Kudo

# Vérifiez que l'environnement virtuel est activé
source .venv/bin/activate
```

### "OPENAI_API_KEY non configurée"

```bash
# Vérifiez que le fichier .env existe
ls -la .env

# Éditez-le et ajoutez votre clé
nano .env  # ou code .env
```

### "Aucun document trouvé"

```bash
# Vérifiez que vos documents sont dans data/raw/
ls -la data/raw/

# Les formats supportés sont: .pdf, .docx, .md, .html
```

### "Impossible de charger l'index"

```bash
# L'index n'existe pas encore, créez-le d'abord
python scripts/pipeline.py index

# Ou pipeline complet si vous n'avez rien fait encore
python scripts/pipeline.py full
```

## Prochaines étapes

Une fois le système fonctionnel :

1. **Enrichir la base** : Ajoutez plus de documents dans `data/raw/`
2. **Tester différentes questions** : Explorez les capacités du système
3. **Affiner la configuration** : Ajustez les paramètres dans `.env`
4. **Évaluer la qualité** : Notez les bonnes et mauvaises réponses
5. **Explorer le code** : Consultez les modules dans `src/`

## Support

Pour plus de détails, consultez :
- [README.md](README.md) - Documentation complète
- [.env.example](.env.example) - Toutes les options de configuration
- Les docstrings dans le code source

Bon entraînement ! 🥋
