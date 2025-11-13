#!/usr/bin/env python3
"""
Script de diagnostic GPU pour vérifier la configuration CUDA.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.gpu_utils import print_gpu_info, get_optimal_batch_size, configure_cuda_optimizations


def check_sentence_transformers():
    """Vérifie que sentence-transformers peut utiliser CUDA."""
    print("\n🔍 Test Sentence Transformers...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        device = model.device

        print(f"✅ Sentence Transformers chargé sur: {device}")

        # Test d'embedding
        test_text = "Test d'embedding avec GPU"
        embedding = model.encode(test_text)
        print(f"✅ Embedding généré: {len(embedding)} dimensions")

    except Exception as e:
        print(f"❌ Erreur: {e}")


def check_docling():
    """Vérifie la configuration Docling."""
    print("\n🔍 Test Docling...")
    try:
        from docling.document_converter import DocumentConverter
        print("✅ Docling importé avec succès")
        print("   Note: Docling utilisera automatiquement CUDA pour l'OCR si disponible")
    except Exception as e:
        print(f"❌ Erreur: {e}")


def main():
    print("\n" + "="*70)
    print("🎮 Diagnostic GPU - RAG-Kudo")
    print("="*70)

    # Info GPU de base
    print_gpu_info()

    # Optimisations CUDA
    if torch.cuda.is_available():
        configure_cuda_optimizations()
        print("✅ Optimisations CUDA activées (cuDNN benchmark + TF32)")

    # Batch sizes recommandés
    print("\n📊 Batch sizes recommandés pour votre GPU:")
    print(f"  - Embeddings: {get_optimal_batch_size('embedding')}")
    print(f"  - OCR: {get_optimal_batch_size('ocr')}")

    # Tests des bibliothèques
    check_sentence_transformers()
    check_docling()

    # Résumé
    print("\n" + "="*70)
    if torch.cuda.is_available():
        print("✅ Configuration GPU: OPÉRATIONNELLE")
        print("\n💡 Conseils:")
        print("  - Vos embeddings locaux seront calculés sur GPU")
        print("  - L'OCR de Docling utilisera le GPU automatiquement")
        print("  - Pour OpenAI embeddings, le GPU n'est pas utilisé (API cloud)")
        print("\n  Ajoutez à votre .env:")
        print("  USE_GPU=true")
        print(f"  EMBEDDING_BATCH_SIZE={get_optimal_batch_size('embedding')}")
        print(f"  OCR_BATCH_SIZE={get_optimal_batch_size('ocr')}")
    else:
        print("⚠️  Pas de GPU CUDA détecté - Fonctionnement en mode CPU")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
