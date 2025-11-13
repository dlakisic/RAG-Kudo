#!/bin/bash

# Script de lancement de l'interface Chainlit

echo "🥋 Lancement de RAG-Kudo Chainlit..."
echo ""

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "app/chainlit_app.py" ]; then
    echo "❌ Erreur: app/chainlit_app.py introuvable"
    echo "Assurez-vous d'être dans le répertoire RAG-Kudo"
    exit 1
fi

# Vérifier que l'environnement virtuel existe
if [ ! -d ".venv" ]; then
    echo "❌ Erreur: Environnement virtuel .venv introuvable"
    echo "Exécutez : uv sync"
    exit 1
fi

# Activer l'environnement virtuel
source .venv/bin/activate

# Vérifier que l'index existe
if [ ! -d "data/vectorstore" ] || [ -z "$(ls -A data/vectorstore)" ]; then
    echo "⚠️  Attention: L'index vectoriel semble vide"
    echo "Créez l'index avec : python scripts/pipeline.py index"
    echo ""
    read -p "Continuer quand même ? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Lancer Chainlit
echo "✅ Lancement de Chainlit sur http://localhost:8000"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter"
echo ""

chainlit run app/chainlit_app.py -w --host 0.0.0.0 --port 8000
