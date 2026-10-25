import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# Unstructured layout elements
from unstructured.documents.elements import (
    Element,
    Title,
    Header,
    NarrativeText,
    ListItem,
    Text,
    CodeSnippet,
    Table,
    Image,
    FigureCaption,
)
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text


def classify_element(element: Element) -> str:
    """
    Classifies an unstructured element into one of the 5 UI categories:
    'title', 'text', 'table', 'image', or 'other'.
    """
    if isinstance(element, (Title, Header)):
        return "title"
    elif isinstance(element, (NarrativeText, ListItem, Text, CodeSnippet)):
        return "text"
    elif isinstance(element, Table):
        return "table"
    elif isinstance(element, (Image, FigureCaption)):
        return "image"
    else:
        return "other"


def parse_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a CSV file into structured table elements using pandas,
    generating both readable markdown and HTML representations.
    """
    df = pd.read_csv(file_path)
    # Generate markdown table string
    table_markdown = df.to_markdown(index=False)
    table_html = df.to_html(index=False, classes="table table-striped")
    
    element_dict = {
        "type": "table",
        "element_class": "Table",
        "text": f"CSV Table from {file_path.name}:\n\n{table_markdown}",
        "metadata": {
            "page_number": 1,
            "filename": file_path.name,
            "rows": len(df),
            "columns": list(df.columns),
            "text_as_html": table_html,
        }
    }
    return [element_dict]


def parse_document(file_path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Partitions a document (PDF, DOCX, TXT, CSV) into layout elements.
    
    Returns:
        elements: List of parsed element dictionaries with metadata.
        stats: Aggregated counts matching the 'Partitioning' screen:
               - text_sections
               - tables
               - images
               - titles_headers
               - other_elements
               - total_elements
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    raw_elements: List[Element] = []

    if extension == ".csv":
        parsed_elements = parse_csv_file(path)
    elif extension == ".pdf":
        try:
            raw_elements = partition_pdf(
                filename=str(path),
                strategy="fast",
                infer_table_structure=True,
            )
        except Exception:
            from unstructured.partition.auto import partition
            raw_elements = partition(filename=str(path))
    elif extension in [".docx", ".doc"]:
        raw_elements = partition_docx(
            filename=str(path),
            infer_table_structure=True,
        )
    elif extension in [".txt", ".md"]:
        raw_elements = partition_text(filename=str(path))
    else:
        from unstructured.partition.auto import partition
        raw_elements = partition(filename=str(path))

    # Convert raw unstructured elements to standardized dictionaries if not CSV
    if extension != ".csv":
        parsed_elements = []
        for el in raw_elements:
            category = classify_element(el)
            page_no = getattr(el.metadata, "page_number", 1) if el.metadata else 1
            table_html = getattr(el.metadata, "text_as_html", None) if el.metadata else None

            parsed_elements.append({
                "type": category,
                "element_class": el.__class__.__name__,
                "text": str(el.text).strip(),
                "metadata": {
                    "page_number": page_no,
                    "filename": path.name,
                    "text_as_html": table_html,
                }
            })

    # Filter out empty text elements
    parsed_elements = [el for el in parsed_elements if el["text"] or el["type"] in ["table", "image"]]

    # Compute discovery statistics matching the UI
    stats = {
        "text_sections": sum(1 for el in parsed_elements if el["type"] == "text"),
        "tables": sum(1 for el in parsed_elements if el["type"] == "table"),
        "images": sum(1 for el in parsed_elements if el["type"] == "image"),
        "titles_headers": sum(1 for el in parsed_elements if el["type"] == "title"),
        "other_elements": sum(1 for el in parsed_elements if el["type"] == "other"),
        "total_elements": len(parsed_elements),
    }

    return parsed_elements, stats
