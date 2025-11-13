"""
Module d'ingestion Docling optimisé pour le multilangue (français, anglais, russe).
Configure RapidOCR avec le modèle multilingue.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from loguru import logger

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrOptions


class MultilingualDoclingProcessor:
    """
    Processeur Docling optimisé pour documents multilingues.
    Supporte français, anglais, russe et autres langues.
    """

    def __init__(
        self,
        output_dir: Path,
        extract_tables: bool = True,
        extract_images: bool = False,
        ocr_enabled: bool = True,
        languages: Optional[List[str]] = None,
    ):
        """
        Initialise le processeur multilingue.

        Args:
            output_dir: Répertoire de sortie
            extract_tables: Extraire les tableaux
            extract_images: Extraire les images
            ocr_enabled: Activer l'OCR
            languages: Langues à détecter (par défaut: ['fr', 'en', 'ru'])
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.languages = languages or ['fr', 'en', 'ru']

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = extract_tables
        pipeline_options.do_ocr = ocr_enabled

        if ocr_enabled:
            pipeline_options.ocr_options = OcrOptions(
                kind="rapidocr",
                force_full_page_ocr=False,
            )

        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.MD,
                InputFormat.HTML,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        logger.info(
            f"MultilingualDoclingProcessor initialisé - "
            f"Langues: {', '.join(self.languages)} - "
            f"Output: {output_dir}"
        )

    def process_document(self, file_path: Path, language_hint: Optional[str] = None) -> Dict:
        """
        Traite un document avec détection automatique de la langue.

        Args:
            file_path: Chemin vers le document
            language_hint: Indice sur la langue du document ('fr', 'en', 'ru', etc.)

        Returns:
            Données du document structuré
        """
        logger.info(f"Traitement du document: {file_path}")
        if language_hint:
            logger.info(f"Langue suggérée: {language_hint}")

        try:
            result = self.converter.convert(str(file_path))

            doc_data = {
                "source_file": str(file_path),
                "file_name": file_path.name,
                "language_hint": language_hint,
                "content": result.document.export_to_markdown(),
                "metadata": {
                    "num_pages": len(result.document.pages) if hasattr(result.document, 'pages') else None,
                    "format": file_path.suffix,
                },
                "structure": self._extract_structure(result),
            }

            output_file = self.output_dir / f"{file_path.stem}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, ensure_ascii=False, indent=2)

            logger.success(f"Document traité: {output_file}")
            return doc_data

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path}: {e}")
            raise

    def _extract_structure(self, result) -> List[Dict]:
        """Extrait la structure du document."""
        structure = []
        try:
            for item in result.document.iterate_items():
                if hasattr(item, 'label') and hasattr(item, 'text'):
                    structure.append({
                        "type": item.label,
                        "text": item.text,
                        "level": getattr(item, 'level', 0),
                    })
        except Exception as e:
            logger.warning(f"Impossible d'extraire la structure: {e}")
        return structure

    def process_directory(
        self,
        input_dir: Path,
        recursive: bool = True,
        language_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Traite tous les documents d'un répertoire.

        Args:
            input_dir: Répertoire des documents
            recursive: Traiter les sous-répertoires
            language_mapping: Mapping fichier -> langue (ex: {"doc_ru.pdf": "ru"})

        Returns:
            Liste des documents traités
        """
        input_dir = Path(input_dir)
        logger.info(f"Traitement du répertoire: {input_dir}")

        language_mapping = language_mapping or {}

        extensions = ['.pdf', '.docx', '.md', '.html']

        if recursive:
            files = [f for ext in extensions for f in input_dir.rglob(f"*{ext}")]
        else:
            files = [f for ext in extensions for f in input_dir.glob(f"*{ext}")]

        logger.info(f"Trouvé {len(files)} fichiers à traiter")

        results = []
        for file_path in files:
            try:
                language_hint = language_mapping.get(file_path.name)
                if not language_hint:
                    filename_lower = file_path.name.lower()
                    if any(x in filename_lower for x in ['ru', 'russian', 'cyrillic']):
                        language_hint = 'ru'
                    elif any(x in filename_lower for x in ['fr', 'french', 'francais']):
                        language_hint = 'fr'
                    else:
                        language_hint = 'en'

                doc_data = self.process_document(file_path, language_hint=language_hint)
                results.append(doc_data)
            except Exception as e:
                logger.error(f"Échec du traitement de {file_path}: {e}")
                continue

        logger.success(f"Traitement terminé: {len(results)}/{len(files)} documents réussis")
        return results


def main():
    """Test du module."""
    from pathlib import Path

    input_dir = Path("data/raw")
    output_dir = Path("data/processed")

    language_mapping = {
        "pravila-vida-sporta-kudo_27.02.24.pdf": "ru",
        "01 L'ARBITRAGE AU KUDO.pdf": "fr",
        "KIF-Tournament-Rules.pdf": "en",
    }

    processor = MultilingualDoclingProcessor(output_dir=output_dir)

    if input_dir.exists():
        results = processor.process_directory(
            input_dir,
            language_mapping=language_mapping
        )
        logger.info(f"Traités {len(results)} documents")
    else:
        logger.warning(f"Le répertoire {input_dir} n'existe pas")


if __name__ == "__main__":
    main()
