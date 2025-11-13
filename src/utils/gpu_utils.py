"""
Utilitaires pour la détection et l'utilisation du GPU.
"""

import torch
from loguru import logger
from typing import Optional


def get_device() -> torch.device:
    """
    Retourne le device optimal (cuda si disponible, sinon cpu).

    Returns:
        Device PyTorch
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU détecté: {gpu_name} ({vram:.1f} GB VRAM)")
        return device
    else:
        logger.warning("Pas de GPU CUDA détecté, utilisation du CPU")
        return torch.device("cpu")


def get_optimal_batch_size(task: str = "embedding") -> int:
    """
    Retourne une taille de batch optimale selon le GPU disponible.

    Args:
        task: Type de tâche ('embedding', 'ocr', etc.)

    Returns:
        Taille de batch recommandée
    """
    if not torch.cuda.is_available():
        return 8

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if task == "embedding":
        if vram_gb >= 12:
            return 64
        elif vram_gb >= 8:
            return 32
        else:
            return 16
    elif task == "ocr":
        if vram_gb >= 12:
            return 4
        elif vram_gb >= 8:
            return 2
        else:
            return 1

    return 16


def clear_gpu_memory():
    """Nettoie la mémoire GPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Mémoire GPU nettoyée")


def get_gpu_stats() -> dict:
    """
    Retourne les statistiques d'utilisation du GPU.

    Returns:
        Dictionnaire avec les stats GPU
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    stats = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "allocated_memory_gb": torch.cuda.memory_allocated(0) / (1024**3),
        "cached_memory_gb": torch.cuda.memory_reserved(0) / (1024**3),
    }

    return stats


def print_gpu_info():
    """Affiche les informations sur le GPU de manière formatée."""
    stats = get_gpu_stats()

    if not stats["cuda_available"]:
        print("❌ Aucun GPU CUDA détecté")
        return

    print("\n" + "="*60)
    print("🎮 Configuration GPU")
    print("="*60)
    print(f"GPU: {stats['device_name']}")
    print(f"VRAM totale: {stats['total_memory_gb']:.2f} GB")
    print(f"VRAM allouée: {stats['allocated_memory_gb']:.2f} GB")
    print(f"VRAM en cache: {stats['cached_memory_gb']:.2f} GB")
    print(f"VRAM disponible: {stats['total_memory_gb'] - stats['allocated_memory_gb']:.2f} GB")
    print("="*60 + "\n")


def configure_cuda_optimizations():
    """Configure les optimisations CUDA pour PyTorch."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        logger.info("Optimisations CUDA configurées (cuDNN benchmark + TF32)")


if __name__ == "__main__":
    print_gpu_info()
    print(f"\nDevice optimal: {get_device()}")
    print(f"Batch size recommandé (embedding): {get_optimal_batch_size('embedding')}")
    print(f"Batch size recommandé (OCR): {get_optimal_batch_size('ocr')}")
