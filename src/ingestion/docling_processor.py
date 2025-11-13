"""
Module d'ingestion de documents utilisant Docling.
Traite les PDFs, Word, Markdown et autres formats pour extraire le contenu structuré.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
from loguru import logger

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


class DoclingProcessor:
    """Processeur de documents utilisant Docling pour l'extraction structurée."""

    def __init__(
        self,
        output_dir: Path,
        extract_tables: bool = True,
        extract_images: bool = False,
        ocr_enabled: bool = True
    ):
        """
        Initialise le processeur Docling.

        Args:
            output_dir: Répertoire de sortie pour les documents traités
            extract_tables: Extraire les tableaux
            extract_images: Extraire les images
            ocr_enabled: Activer l'OCR pour les documents scannés
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = extract_tables
        pipeline_options.do_ocr = ocr_enabled

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

        logger.info(f"DoclingProcessor initialisé avec output_dir: {output_dir}")

    def process_document(self, file_path: Path) -> Dict:
        """
        Traite un document avec Docling.

        Args:
            file_path: Chemin vers le document à traiter

        Returns:
            Dict contenant le document structuré et les métadonnées
        """
        logger.info(f"Traitement du document: {file_path}")

        try:
            result = self.converter.convert(str(file_path))

            doc_data = {
                "source_file": str(file_path),
                "file_name": file_path.name,
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

            logger.success(f"Document traité avec succès: {output_file}")
            return doc_data

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path}: {e}")
            raise

    def _extract_structure(self, result) -> List[Dict]:
        """
        Extrait la structure hiérarchique du document.

        Args:
            result: Résultat de conversion Docling

        Returns:
            Liste des sections structurées
        """
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
            logger.warning(f"Impossible d'extraire la structure détaillée: {e}")

        return structure

    def process_directory(self, input_dir: Path, recursive: bool = True) -> List[Dict]:
        """
        Traite tous les documents d'un répertoire.

        Args:
            input_dir: Répertoire contenant les documents
            recursive: Traiter les sous-répertoires

        Returns:
            Liste des documents traités
        """
        input_dir = Path(input_dir)
        logger.info(f"Traitement du répertoire: {input_dir}")

        extensions = ['.pdf', '.docx', '.md', '.html']

        if recursive:
            files = [f for ext in extensions for f in input_dir.rglob(f"*{ext}")]
        else:
            files = [f for ext in extensions for f in input_dir.glob(f"*{ext}")]

        logger.info(f"Trouvé {len(files)} fichiers à traiter")

        results = []
        for file_path in files:
            try:
                doc_data = self.process_document(file_path)
                results.append(doc_data)
            except Exception as e:
                logger.error(f"Échec du traitement de {file_path}: {e}")
                continue

        logger.success(f"Traitement terminé: {len(results)}/{len(files)} documents réussis")
        return results

    def extract_sections_by_type(self, doc_data: Dict, section_type: str) -> List[str]:
        """
        Extrait les sections d'un type spécifique.

        Args:
            doc_data: Données du document traité
            section_type: Type de section à extraire (ex: "heading", "paragraph", "table")

        Returns:
            Liste des textes des sections du type demandé
        """
        sections = []
        for item in doc_data.get("structure", []):
            if item.get("type") == section_type:
                sections.append(item.get("text", ""))

        return sections


def main():
    """Fonction de test du module."""
    from pathlib import Path

    input_dir = Path("data/raw")
    output_dir = Path("data/processed")

    processor = DoclingProcessor(output_dir=output_dir)

    if input_dir.exists():
        results = processor.process_directory(input_dir)
        logger.info(f"Traités {len(results)} documents")
    else:
        logger.warning(f"Le répertoire {input_dir} n'existe pas")


if __name__ == "__main__":
    main()
