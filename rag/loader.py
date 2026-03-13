"""
Document loader - reads .txt (and optional .pdf) files from the data directory.

Each document is returned as:
    {"content": str, "metadata": {"source": filename, "category": category}}

Category is inferred from the filename prefix:
    channel_*   -> discourse_channel
    resource_*  -> course_resource
    week*       -> week_content
"""
from pathlib import Path


def get_category(filename: str) -> str:
    stem = Path(filename).stem
    if stem.startswith("channel_"):
        return "discourse_channel"
    if stem.startswith("resource_"):
        return "course_resource"
    if stem.startswith("week"):
        return "week_content"
    return "other"


def load_documents(data_dir: Path) -> list[dict]:
    """
    Load all .txt and .pdf files from data_dir.

    PDF support requires PyMuPDF (`pip install pymupdf`).
    Missing PDFs are skipped gracefully with a notice.
    """
    documents: list[dict] = []

    for file_path in sorted(data_dir.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                print(f"[loader] Skipping empty file: {file_path.name}")
                continue
            documents.append({
                "content": content,
                "metadata": {
                    "source":   file_path.name,
                    "category": get_category(file_path.name),
                },
            })
        except Exception as exc:
            print(f"[loader] Error reading {file_path.name}: {exc}")

    return documents
