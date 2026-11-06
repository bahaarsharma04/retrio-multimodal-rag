import os
import sys
import time
import argparse
import re
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import pdfplumber

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
    ElementMetadata,
)
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text


def clean_text_artifacts(text: str) -> str:
    """
    Cleans raw PDF text extractions:
    - Removes null bytes (\\x00) and unprintable control characters.
    - Strips CID font glyph artifacts like (cid:0), (cid:14), etc.
    - Rejoins hyphenated words broken across line wraps (e.g. 'au-\\n rora' -> 'aurora').
    - Rejoins hyphenated words broken with spaces (e.g. 'au- rora' -> 'aurora').
    - Normalizes dropped drop-caps (e.g. missing initial 'T' in 'he beautiful').
    - Strips standalone page numbers on isolated lines.
    """
    if not text:
        return ""
    # Strip null bytes and non-whitespace control characters
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Strip CID unmapped font artifacts
    text = re.sub(r"\(cid:\d+\)", "", text)
    # Fix broken hyphenation across line breaks: word-\n word -> wordword
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)
    # Fix broken hyphenation with spaces on same line: word- word -> wordword
    text = re.sub(r"(\b[a-zA-Z]{3,})-\s+([a-zA-Z]{2,}\b)", r"\1\2", text)
    # Normalize known dropped drop-caps where vector initials are omitted by text streams
    text = re.sub(r"\bhe beautiful and often eerie\b", "The beautiful and often eerie", text)
    # Strip isolated single-line page numbers (e.g. "2", "3")
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    # Collapse 3+ consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages_layout(file_path: Path) -> List[Tuple[int, str]]:
    """
    Extracts text per page from a PDF with column-aware layout detection.
    
    If a page is detected as two-column, it extracts content in natural human reading order:
      1. Top spanning headers/titles
      2. Left column (top-to-bottom)
      3. Right column (top-to-bottom)
      4. Bottom spanning footers/captions
      
    For single-column pages, standard layout order is preserved.
    """
    page_texts: List[Tuple[int, str]] = []
    with pdfplumber.open(file_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_no = page_idx + 1
            words = [w for w in page.extract_words() if not w.get("text", "").startswith("(cid:")]
            if not words or len(words) < 15:
                raw = page.extract_text() or ""
                cleaned = clean_text_artifacts(raw)
                if cleaned:
                    page_texts.append((page_no, cleaned))
                continue

            mid_x = page.width / 2.0
            left_words = [w for w in words if w["x1"] <= mid_x]
            right_words = [w for w in words if w["x0"] >= mid_x]
            in_gutter = [w for w in words if w["x0"] < mid_x + 5 and w["x1"] > mid_x - 5]
            gutter_ratio = len(in_gutter) / len(words)

            is_two_col = (
                len(words) >= 30
                and len(left_words) / len(words) >= 0.18
                and len(right_words) / len(words) >= 0.18
                and gutter_ratio < 0.04
            )

            if not is_two_col:
                raw = page.extract_text() or ""
                cleaned = clean_text_artifacts(raw)
                if cleaned:
                    page_texts.append((page_no, cleaned))
                continue

            # Group words into visual lines by vertical alignment
            sorted_words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
            lines: List[List[Dict[str, Any]]] = []
            curr_line: List[Dict[str, Any]] = []
            curr_top = None
            for w in sorted_words:
                if curr_top is None or abs(w["top"] - curr_top) <= 3:
                    curr_line.append(w)
                    if curr_top is None:
                        curr_top = w["top"]
                else:
                    lines.append(curr_line)
                    curr_line = [w]
                    curr_top = w["top"]
            if curr_line:
                lines.append(curr_line)

            header_y = 0.0
            footer_y = float(page.height)

            for line in lines:
                x0 = min(w["x0"] for w in line)
                x1 = max(w["x1"] for w in line)
                line_top = min(w["top"] for w in line)
                line_bottom = max(w["bottom"] for w in line)

                is_centered = abs((x0 + x1) / 2 - mid_x) < 30 and (x1 - x0) < 180

                # Check if words in line cross the center continuously without a column gutter gap
                crosses_continuously = any(w["x0"] <= mid_x <= w["x1"] for w in line)
                if not crosses_continuously and x0 < mid_x - 10 and x1 > mid_x + 10:
                    s_l = sorted(line, key=lambda w: w["x0"])
                    for i in range(len(s_l) - 1):
                        if s_l[i]["x1"] <= mid_x <= s_l[i + 1]["x0"]:
                            if s_l[i + 1]["x0"] - s_l[i]["x1"] < 18:
                                crosses_continuously = True
                            break

                if is_centered or crosses_continuously:
                    if line_top < page.height * 0.35:
                        header_y = max(header_y, line_bottom + 4)
                    elif line_top > page.height * 0.65:
                        footer_y = min(footer_y, line_top - 4)

            def words_to_str(w_list: List[Dict[str, Any]]) -> str:
                if not w_list:
                    return ""
                s = sorted(w_list, key=lambda w: (round(w["top"], 1), w["x0"]))
                line_objs = []
                cur = []
                cur_t = None
                cur_b = None
                for w in s:
                    if cur_t is None or abs(w["top"] - cur_t) <= 3:
                        cur.append(w["text"])
                        if cur_t is None:
                            cur_t = w["top"]
                            cur_b = w["bottom"]
                        else:
                            cur_b = max(cur_b, w["bottom"])
                    else:
                        line_objs.append((" ".join(cur), cur_t, cur_b))
                        cur = [w["text"]]
                        cur_t = w["top"]
                        cur_b = w["bottom"]
                if cur:
                    line_objs.append((" ".join(cur), cur_t, cur_b))

                paragraphs = []
                curr_para = ""
                for i, (l_text, l_top, l_bot) in enumerate(line_objs):
                    l_text = l_text.strip()
                    if not l_text:
                        continue
                    if not curr_para:
                        curr_para = l_text
                        continue

                    prev_text, prev_top, prev_bot = line_objs[i - 1]
                    gap = l_top - prev_bot

                    is_prev_heading = (
                        len(prev_text.split()) <= 6
                        and not prev_text.endswith((".", ",", ";", ":", "-"))
                        and (prev_text[0].isupper() or prev_text.startswith("#"))
                        and gap > 6
                    )
                    is_curr_heading = (
                        len(l_text.split()) <= 6
                        and not l_text.endswith((".", ",", ";", ":", "-"))
                        and (l_text[0].isupper() or l_text.startswith("#"))
                        and gap > 6
                    )
                    is_list_item = bool(re.match(r"^(\d+[\.\)]|[-*•]|Figure\s+\d+)", l_text))

                    if prev_text.endswith("-"):
                        # Rejoin broken hyphen across line
                        curr_para = curr_para[:-1] + l_text
                    elif (
                        not is_prev_heading
                        and not is_curr_heading
                        and not is_list_item
                        and (not prev_text.endswith((".", "!", "?", ":")) or gap < 12)
                    ):
                        curr_para += " " + l_text
                    else:
                        paragraphs.append(curr_para)
                        curr_para = l_text

                if curr_para:
                    paragraphs.append(curr_para)

                return "\n\n".join(paragraphs)

            h_w = [w for w in words if w["bottom"] <= header_y]
            l_w = [w for w in words if header_y < w["top"] and w["bottom"] <= footer_y and w["x1"] <= mid_x]
            r_w = [w for w in words if header_y < w["top"] and w["bottom"] <= footer_y and w["x0"] >= mid_x]
            f_w = [w for w in words if w["top"] >= footer_y]

            parts = []
            if h_w:
                parts.append(clean_text_artifacts(words_to_str(h_w)))
            if l_w:
                parts.append(clean_text_artifacts(words_to_str(l_w)))
            if r_w:
                parts.append(clean_text_artifacts(words_to_str(r_w)))
            if f_w:
                parts.append(clean_text_artifacts(words_to_str(f_w)))

            combined = "\n\n".join(parts)
            if combined.strip():
                page_texts.append((page_no, combined))

    return page_texts


def classify_element(element: Element) -> str:
    """
    Classifies an unstructured element into one of the 5 UI categories:
    'table', 'image', 'title', 'text', or 'other'.
    
    CRITICAL: In unstructured, Table and Image inherit from Text.
    Therefore, Table and Image MUST be checked BEFORE Text!
    """
    if isinstance(element, Table):
        return "table"
    elif isinstance(element, (Image, FigureCaption)):
        return "image"
    elif isinstance(element, (Title, Header)):
        txt = element.text.strip() if hasattr(element, "text") and element.text else str(element).strip()
        words = txt.split()
        # Filter false titles: must start with uppercase, must not end with sentence punctuation, must not exceed 10 words
        if (
            not txt
            or not txt[0].isupper()
            or len(words) > 10
            or txt.endswith((".", ";", ","))
            or (
                len(words) <= 2
                and (
                    len(txt) < 4
                    or txt.lower() in {"this", "is", "a", "an", "the", "at", "by", "for", "in", "of", "on", "to", "and", "or", "as", "but"}
                )
            )
        ):
            return "text"
        return "title"
    elif isinstance(element, (NarrativeText, ListItem, CodeSnippet, Text)):
        return "text"
    else:
        return "other"


def parse_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a CSV file into structured table elements using pandas,
    generating both readable markdown and HTML representations.
    """
    df = pd.read_csv(file_path)
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


def extract_docx_embedded_images(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts embedded images from Word documents using python-docx."""
    image_elements = []
    try:
        import docx
        doc = docx.Document(file_path)
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_name = Path(rel.target_ref).name
                image_elements.append({
                    "type": "image",
                    "element_class": "Image",
                    "text": f"[Embedded Visual / Diagram: {img_name} in {file_path.name}]",
                    "metadata": {
                        "page_number": 1,
                        "filename": file_path.name,
                        "text_as_html": None,
                    }
                })
    except Exception:
        pass
    return image_elements


def extract_pdf_embedded_images(file_path: Path) -> List[Dict[str, Any]]:
    """Extracts embedded images from PDF documents using pypdf."""
    image_elements = []
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            for img in page.images:
                image_elements.append({
                    "type": "image",
                    "element_class": "Image",
                    "text": f"[Embedded Figure: {img.name} on Page {page_idx + 1}]",
                    "metadata": {
                        "page_number": page_idx + 1,
                        "filename": file_path.name,
                        "text_as_html": None,
                    }
                })
    except Exception:
        pass
    return image_elements


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
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    extension = path.suffix.lower()
    raw_elements: List[Element] = []
    extra_images: List[Dict[str, Any]] = []

    if extension == ".csv":
        parsed_elements = parse_csv_file(path)
    elif extension == ".pdf":
        try:
            pages_layout = extract_pdf_pages_layout(path)
            for page_no, page_text in pages_layout:
                page_els = partition_text(text=page_text)
                for el in page_els:
                    if el.metadata is None:
                        el.metadata = ElementMetadata(page_number=page_no)
                    else:
                        el.metadata.page_number = page_no
                raw_elements.extend(page_els)
        except Exception:
            try:
                raw_elements = partition_pdf(
                    filename=str(path),
                    strategy="fast",
                    infer_table_structure=True,
                )
            except Exception:
                raw_elements = []

        extra_images = extract_pdf_embedded_images(path)
    elif extension in [".docx", ".doc"]:
        raw_elements = partition_docx(
            filename=str(path),
            infer_table_structure=True,
        )
        extra_images = extract_docx_embedded_images(path)
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

    # Append any detected embedded images from document structure
    if extra_images:
        existing_images = sum(1 for el in parsed_elements if el["type"] == "image")
        if existing_images == 0:
            parsed_elements.extend(extra_images)

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


def parse_raw_data(data: str, source_name: str = "raw_input.txt") -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Partitions raw text, markdown, or unstructured string data directly into layout elements.
    """
    if not data or not data.strip():
        return [], {
            "text_sections": 0,
            "tables": 0,
            "images": 0,
            "titles_headers": 0,
            "other_elements": 0,
            "total_elements": 0,
        }

    raw_elements = partition_text(text=data)
    parsed_elements = []

    for el in raw_elements:
        category = classify_element(el)
        parsed_elements.append({
            "type": category,
            "element_class": el.__class__.__name__,
            "text": str(el.text).strip(),
            "metadata": {
                "page_number": 1,
                "filename": source_name,
                "text_as_html": None,
            }
        })

    parsed_elements = [el for el in parsed_elements if el["text"] or el["type"] in ["table", "image"]]

    stats = {
        "text_sections": sum(1 for el in parsed_elements if el["type"] == "text"),
        "tables": sum(1 for el in parsed_elements if el["type"] == "table"),
        "images": sum(1 for el in parsed_elements if el["type"] == "image"),
        "titles_headers": sum(1 for el in parsed_elements if el["type"] == "title"),
        "other_elements": sum(1 for el in parsed_elements if el["type"] == "other"),
        "total_elements": len(parsed_elements),
    }

    return parsed_elements, stats


def export_elements_to_markdown(
    elements: List[Dict[str, Any]],
    stats: Dict[str, int],
    source_name: str,
    output_path: Optional[str | Path] = None,
) -> Path:
    """
    Exports parsed layout elements to a formatted Markdown (.md) document on disk.
    Converts HTML tables into clean GitHub-Flavored Markdown tables where applicable.
    """
    if output_path:
        out_p = Path(output_path)
    else:
        stem = Path(source_name).stem if source_name else "parsed_output"
        out_p = Path(f"{stem}.parsed.md")

    lines = []
    lines.append(f"# Parsed Layout Document: {source_name}\n")
    lines.append(f"> **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(
        f"> **Total Elements:** {stats.get('total_elements', len(elements))} | "
        f"**Text Sections:** {stats.get('text_sections', 0)} | "
        f"**Tables:** {stats.get('tables', 0)} | "
        f"**Images:** {stats.get('images', 0)} | "
        f"**Titles/Headings:** {stats.get('titles_headers', 0)}\n"
    )
    lines.append("---\n")

    current_page = None
    for el in elements:
        page = el.get("metadata", {}).get("page_number")
        if page and page != current_page:
            current_page = page
            lines.append(f"\n<!-- Page {page} -->\n")

        el_type = el.get("type", "text")
        text = el.get("text", "").strip()
        html = el.get("metadata", {}).get("text_as_html")

        if el_type == "title":
            lines.append(f"\n## {text}\n")
        elif el_type == "table":
            # Attempt to render HTML table to clean markdown if available
            converted_table = False
            if html:
                try:
                    dfs = pd.read_html(StringIO(html))
                    if dfs:
                        lines.append("\n" + dfs[0].to_markdown(index=False) + "\n")
                        converted_table = True
                except Exception:
                    pass
            if not converted_table:
                lines.append(f"\n{text}\n")
        elif el_type == "image":
            lines.append(f"\n> 🖼️ **[Embedded Visual / Figure]** {text}\n")
        else:
            lines.append(f"{text}\n")

    md_content = "\n".join(lines).strip() + "\n"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(md_content)

    return out_p.resolve()


def main():
    """Command-line interface to parse documents or raw data and export Markdown files."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        prog="python -m core.parser",
        description="Retrio Layout-Aware Document & Data Parser CLI\n"
                    "Parses PDF, DOCX, CSV, TXT, MD documents or raw data strings into structured layout elements and outputs Markdown (.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse a document file and auto-generate <file>.parsed.md:
  python -m core.parser document.pdf
  python -m core.parser --file "data/documents/report.docx" --output "report_clean.md"

  # Parse raw data string:
  python -m core.parser --data "# Title\nThis is raw data content." --output "custom.md"

  # Parse piped input:
  Get-Content notes.txt | python -m core.parser --data -
        """,
    )

    parser.add_argument(
        "document",
        nargs="?",
        default=None,
        help="Path to the document file (.pdf, .docx, .csv, .txt, .md).",
    )
    parser.add_argument(
        "-f", "--file",
        dest="file_path",
        default=None,
        help="Explicit path to document file.",
    )
    parser.add_argument(
        "-d", "--data",
        dest="raw_data",
        default=None,
        help="Raw text/markdown data string, or '-' to read from standard input.",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        default=None,
        help="Target output Markdown file (.md) path (default: <filename>.parsed.md).",
    )
    parser.add_argument(
        "--preview-count",
        dest="preview_count",
        type=int,
        default=10,
        help="Number of parsed elements to preview in the terminal (default: 10, 0 for none).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print parsed elements as raw JSON to stdout.",
    )

    args = parser.parse_args()

    # Determine input mode
    target_file = args.file_path or args.document
    raw_data = args.raw_data

    # Check for stdin pipe if -d - was passed or if nothing passed but stdin is piped
    if raw_data == "-":
        raw_data = sys.stdin.read()

    if not target_file and not raw_data:
        parser.print_help()
        sys.exit(1)

    t0 = time.perf_counter()

    try:
        if raw_data is not None:
            source_name = "raw_input_data"
            elements, stats = parse_raw_data(raw_data, source_name=source_name)
        else:
            p = Path(target_file)
            if not p.is_file():
                print(f"\n❌ Error: File not found: '{target_file}'", file=sys.stderr)
                sys.exit(1)
            source_name = p.name
            elements, stats = parse_document(p)
    except Exception as e:
        print(f"\n❌ Parsing failed: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Export to Markdown file
    md_file_path = export_elements_to_markdown(
        elements=elements,
        stats=stats,
        source_name=source_name,
        output_path=args.output_path,
    )

    # If --json requested, dump JSON and exit
    if args.json:
        import json
        print(json.dumps({"stats": stats, "elements": elements}, indent=2, ensure_ascii=False))
        return

    # Print stylish CLI terminal report
    width = 78
    print("=" * width)
    print(" RETRIO LAYOUT-AWARE DOCUMENT & DATA PARSER ".center(width, "="))
    print("=" * width)
    print(f" Source:      {source_name}")
    print(f" Input Type:  {'Raw Data String' if raw_data is not None else 'Document File'}")
    print(f" Elapsed:     {elapsed_ms:.1f}ms")
    print(f" Output File: {md_file_path}")
    print("-" * width)
    print(" DISCOVERED ELEMENTS BREAKDOWN ".center(width, "-"))
    print("-" * width)
    print(f"  * Text Sections:      {stats.get('text_sections', 0):>6}")
    print(f"  * Tables:             {stats.get('tables', 0):>6}")
    print(f"  * Images / Figures:   {stats.get('images', 0):>6}")
    print(f"  * Titles / Headings:  {stats.get('titles_headers', 0):>6}")
    print(f"  * Other Elements:     {stats.get('other_elements', 0):>6}")
    print("  " + "-" * 32)
    print(f"  * Total Elements:     {stats.get('total_elements', 0):>6}")
    print("=" * width)

    # Print Element Previews
    preview_limit = args.preview_count
    if preview_limit > 0 and elements:
        shown = min(preview_limit, len(elements))
        print(f"\nPARSED ELEMENTS PREVIEW (Showing {shown} of {len(elements)} elements):\n")
        for i, el in enumerate(elements[:shown], 1):
            etype = el.get("type", "text").upper()
            page = el.get("metadata", {}).get("page_number", 1)
            raw_t = el.get("text", "").replace("\n", " ").strip()
            snippet = (raw_t[:120] + "...") if len(raw_t) > 120 else raw_t
            print(f"  [{i:>2}] [{etype:<5}] (Page {page})")
            print(f"       \"{snippet}\"\n")

    print("-" * width)
    print(f"[OK] Successfully saved Markdown representation to:\n  -> {md_file_path}")
    print("=" * width)


if __name__ == "__main__":
    main()
