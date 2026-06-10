"""File upload and vision processing — shared by API and admin routes."""
import asyncio
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from fastapi import HTTPException
from pypdf import PdfReader

from config import settings
from rag import chunk_text

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Thread pool for CPU-bound PDF parsing. Size 2 prevents saturating the
# single uvicorn worker while allowing some parallelism for concurrent uploads.
_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="upload")

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Patterns browsers add to duplicate downloads: (1), (2), _copy, _2, etc.
_BROWSER_DUPE_PARENS_RE = re.compile(r"\s*\(\d+\)\s*$")        # " (1)" at end
_BROWSER_DUPE_COPY_RE = re.compile(r"_copy\d*$", re.IGNORECASE)  # "_copy" or "_copy2"
_BROWSER_DUPE_NUM_RE = re.compile(r"_(\d{1,2})$")            # "_2" at end (Chrome), limit to 2 digits to avoid stripping years


def normalize_source_name(filename: str) -> str:
    """Normalize a filename for use as source in the chunks table.

    Strips browser-added duplicate suffixes like (1), (2), _copy, _2
    that appear when downloading a file that already exists locally.
    Also lowercases to prevent casing-based duplicates.
    """
    name = filename.lower()
    # Strip browser duplicate suffixes before the extension
    base, dot_ext = name.rsplit(".", 1) if "." in name else (name, "")
    cleaned = _BROWSER_DUPE_PARENS_RE.sub("", base)
    cleaned = _BROWSER_DUPE_COPY_RE.sub("", cleaned)
    cleaned = _BROWSER_DUPE_NUM_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return f"{cleaned}.{dot_ext}" if dot_ext else cleaned

_VISION_DESCRIBE_PROMPT = (
    "Describe this image in complete detail including all visible text, numbers, labels, "
    "layout, and content. Be exhaustive."
)


def detect_mime(content: bytes, fallback: str = "image/jpeg") -> str:
    """Sniff MIME type from bytes, falling back to a default."""
    try:
        import filetype as _ft
        kind = _ft.guess(content)
        return kind.mime if kind else fallback
    except Exception:
        return fallback


async def describe_image_for_upload(content: bytes, filename: str) -> str:
    """Call vision model to describe an image.

    Returns description text.
    Raises HTTPException 422 if description is empty or too short.
    """
    import base64 as _b64
    from llm import call_chat

    mime = detect_mime(content)
    b64 = _b64.b64encode(content).decode("utf-8")
    vision_model = settings.llm_vision_model or None

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _VISION_DESCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }
    ]
    try:
        description = await call_chat(messages, max_tokens=1000, temperature=0.1, model=vision_model)
    except RuntimeError as e:
        raise HTTPException(422, f"Vision model failed to describe image: {e}")

    # Guard against None (content-filtered images) and very short descriptions
    if not description:
        raise HTTPException(422, "No se pudo describir la imagen. El modelo no pudo procesarla.")

    if len(description.strip()) < 100:
        raise HTTPException(422, "No se pudo extraer contenido de la imagen. Verificá que el modelo soporte visión.")

    return description


def process_uploaded_file(content: bytes, filename: str, source_name: str) -> tuple[list[dict], int, str]:
    """Parse uploaded file content into chunks.

    Returns (chunks, pages_processed, full_doc_text).
    full_doc_text is passed to index_chunks() for contextual retrieval embedding.
    Raises HTTPException on unsupported format, corrupt PDF, etc.
    """
    fname = filename.lower()
    if not (fname.endswith(".pdf") or fname.endswith(".md") or fname.endswith(".txt")):
        raise HTTPException(400, "Supported formats: PDF, Markdown (.md), plain text (.txt)")

    all_chunks = []
    full_doc_text = ""
    if fname.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}")
        page_texts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(page_text)
                chunks = chunk_text(page_text, source=source_name, page=page_num + 1)
                all_chunks.extend(chunks)
        full_doc_text = "\n\n".join(page_texts)
        pages_processed = len(reader.pages)
    else:
        try:
            full_doc_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "File must be UTF-8 encoded")
        if full_doc_text.strip():
            all_chunks = chunk_text(full_doc_text, source=source_name, page=1)
        pages_processed = 1

    if not all_chunks:
        raise HTTPException(400, "No text could be extracted from this file")

    return all_chunks, pages_processed, full_doc_text


async def process_upload_async(content: bytes, filename: str) -> tuple[list[dict], int, str | None]:
    """Async wrapper: offload CPU-bound PDF/text parsing to thread pool.

    Images are described by the vision model (network-bound, stays async).
    PDFs and text files run in the executor to avoid blocking the event loop.
    Returns (chunks, pages_processed, full_doc_text).
    """
    source_name = normalize_source_name(filename)
    fname_lower = filename.lower()
    if any(fname_lower.endswith(ext) for ext in _IMAGE_EXTS):
        description = await describe_image_for_upload(content, fname_lower)
        chunks = chunk_text(description, source=source_name, page=1)
        return chunks, 1, None
    loop = asyncio.get_running_loop()
    chunks, pages, full_doc_text = await loop.run_in_executor(
        _upload_executor, process_uploaded_file, content, fname_lower, source_name,
    )
    return chunks, pages, full_doc_text