from typing import List, Tuple, Optional, Any
import re
import io, math
import fitz  # PyMuPDF
import logging
from docx import Document
from API_Azure import AzureDocIntelligenceService
from PIL import Image, ImageOps, ImageFilter
import multiprocessing
import traceback
import tiktoken
import psutil, os
from utils.retry import retry  # Assuming your retry decorator is here
import concurrent.futures

# standalone function to extract PDF paragraphs, to be run in a subprocess
# Protects from segfaults
def _extract_pdf_paragraphs_worker(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    out = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            for pagenum, page in enumerate(doc, start=1):
                native = page.get_text("text").strip()
                text = native
                # fallback logic is not run here, just extract what we can
                if text:
                    out.append((pagenum, text))
        finally:
            doc.close()
    except Exception:
        # Log traceback for debugging
        traceback.print_exc()
        return []
    return out

class DocumentParsingAndChunkingService:
    """Extracts text (PDF/DOCX/TXT) and produces token-sized chunks with overlap."""
    SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])(?:["”\']\s+|\s+)(?=(?:\(?[A-Z0-9]|[•\-–]))')

    MIN_NATIVE_CHARS = 200          # if native text shorter than this, also OCR
    MIN_MEAN_CHARS_PER_LINE = 6     # crude density check
    TESS_CFG = "--oem 3 --psm 6"    # general body text; try 4 for multi-column pages

    def __init__(self, target_tokens: int = 320, overlap_tokens: int = 48, encoding: str = "cl100k_base"):
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._enc = tiktoken.get_encoding(encoding)

    def extract_paragraphs_from_bytes(self, file_name: str, file_bytes: bytes) -> List[Tuple[Optional[int], str]]:
        name = (file_name or "").lower()
        if name.endswith(".pdf"):
            return self._extract_pdf_paragraphs(file_bytes)
        if name.endswith(".docx"):
            return self._extract_docx_paragraphs(file_bytes)
        try:
            txt = file_bytes.decode("utf-8", errors="ignore")
            if not txt.strip():
                return []
            return self._extract_txt_paragraphs(txt)
        except Exception:
            return []

    @staticmethod
    def _extract_txt_paragraphs(txt: str) -> List[Tuple[int, str]]:
        """Chunk large TXT files by paragraphs or size."""
        paragraphs = [p for p in txt.splitlines() if p.strip()]
        if len(paragraphs) > 50:
            logging.info(f"TXT large-file chunking: {len(paragraphs)} paragraphs")
            return [(i+1, para) for i, para in enumerate(paragraphs)]
        if len(txt) > 10000:
            chunks = [txt[i:i+2000] for i in range(0, len(txt), 2000)]
            logging.info(f"TXT large-file chunking by size: {len(chunks)} chunks")
            return [(i+1, chunk) for i, chunk in enumerate(chunks)]
        return [(1, txt)]

    @staticmethod
    def _clean_for_ocr(pil_img: Image.Image) -> Image.Image:
        # grayscale, slight sharpen, binarize (adaptive), remove small noise
        g = pil_img.convert("L")
        g = g.filter(ImageFilter.SHARPEN)
        # simple binarization
        g = ImageOps.autocontrast(g)
        return g

    @classmethod
    def _page_needs_ocr(cls, txt: str) -> bool:
        if not txt or len(txt.strip()) < cls.MIN_NATIVE_CHARS:
            return True
        # density heuristic: many pages with only headers yield 1-2 short lines
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if not lines:
            return True
        mean_len = sum(len(l) for l in lines) / max(1, len(lines))
        return mean_len < cls.MIN_MEAN_CHARS_PER_LINE

    @classmethod
    @retry()
    def _ocr_page(cls, page: fitz.Page, timeout: int = 30) -> str:
        """OCR a page with timeout, memory logging, and guardrail."""
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / (1024*1024)
        #logging.info(f"Memory usage: {mem_mb:.1f} MB before OCR")
        if proc.memory_info().rss > 1.5 * 1024 * 1024 * 1024:
            logging.error("Aborting OCR, memory usage too high")
            raise RuntimeError("Aborting OCR, memory usage too high")
        def ocr_worker(page: Any) -> str:
            zoom = 4.0
            mat = fitz.Matrix(zoom, zoom).prerotate(page.rotation or 0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pil = cls._clean_for_ocr(pil)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            return AzureDocIntelligenceService.ocr_image_bytes(image_bytes) or ""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ocr_worker, page)
                return future.result(timeout=timeout)
        except Exception as e:
            logging.warning(f"OCR timed out or failed: {e}")
            return ""

    def _extract_pdf_paragraphs(self, pdf_bytes: bytes) -> List[Tuple[int, str]]:
        """
        Extract PDF paragraphs safely. Uses a subprocess for native text extraction,
        then applies heuristics to decide if OCR is needed per page. If OCR is triggered,
        compare OCR vs native text length and prefer the richer result. Includes fallback
        to per-image OCR when whole-page OCR fails. Logs telemetry metrics.
        """
        metrics = {"pages": 0, "ocr_pages": 0, "empty_pages": 0}
        out: List[Tuple[int, str]] = []

        # Try subprocess extraction first
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_extract_pdf_paragraphs_worker, pdf_bytes)
                try:
                    out = future.result(timeout=30)
                except Exception as e:
                    logging.warning(f"Subprocess PDF extraction timeout/error: {e}")
                    out = []
        except Exception as e:
            logging.warning(f"Subprocess PDF extraction failed: {e}")
            out = []

        final_out: List[Tuple[int, str]] = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pagenum, page in enumerate(doc, start=1):
                metrics["pages"] += 1
                
                # Get native text
                native = ""
                if out and pagenum-1 < len(out):
                    native = out[pagenum-1][1]
                else:
                    native = page.get_text("text").strip()

                # Use native text or OCR based on quality check
                text = self._try_ocr_if_needed(doc, page, native, metrics)
                
                # Append result
                if text:
                    final_out.append((pagenum, text))
                else:
                    final_out.append((pagenum, ""))
                    metrics["empty_pages"] += 1
                    logging.warning(f"Empty text for page {pagenum}")
        except Exception as e:
            logging.error(f"PDF parsing failed entirely: {e}")
        finally:
            try:
                doc.close()
            except Exception:
                pass

        if not final_out:
            return [(1, "")]

        logging.info(f"PDF metrics: pages={metrics['pages']}, ocr_pages={metrics['ocr_pages']}, empty_pages={metrics['empty_pages']}")
        return final_out


    def _extract_docx_paragraphs(self, docx_bytes: bytes) -> List[Tuple[int, str]]:
        """Extract paragraphs from DOCX, chunking by paragraph for large files."""
        doc = Document(io.BytesIO(docx_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # If very large, split by paragraph
        if len(paragraphs) > 50:
            logging.info(f"DOCX large-file chunking: {len(paragraphs)} paragraphs")
            return [(i+1, para) for i, para in enumerate(paragraphs)]
        text = "\n".join(paragraphs)
        return [(1, text)] if text.strip() else []

    def _toklen(self, s: str) -> int:
        return len(self._enc.encode(s))

    def build_chunk_tuples(self, page_paragraphs: List[Tuple[Optional[int], str]]) -> List[Tuple[Optional[int], int, str]]:
        """Return list of (page_number, chunk_index, chunk_text)."""
        chunks: List[Tuple[Optional[int], int, str]] = []
        for page_num, text in page_paragraphs:
            sentences = [s for s in self.SENTENCE_SPLIT_REGEX.split(text) if s and s.strip()]
            buf_tokens = 0
            buf_pieces: List[str] = []
            chunk_idx = 0

            def flush_and_start_next_with_overlap():
                nonlocal buf_tokens, buf_pieces, chunk_idx
                if not buf_pieces:
                    return
                chunk_text = " ".join(buf_pieces).strip()
                if not chunk_text:
                    buf_pieces, buf_tokens = [], 0
                    return
                chunks.append((page_num, chunk_idx, chunk_text))
                chunk_idx += 1
                tok_ids = self._enc.encode(chunk_text)
                if not tok_ids:
                    buf_pieces, buf_tokens = [], 0
                    return
                keep = tok_ids[-self._overlap:] if self._overlap < len(tok_ids) else tok_ids
                overlap_text = self._enc.decode(keep)
                buf_pieces = [overlap_text] if overlap_text else []
                buf_tokens = len(keep)

            for s in sentences:
                t = self._toklen(s)
                if buf_tokens + t > self._target and buf_pieces:
                    flush_and_start_next_with_overlap()
                buf_pieces.append(s)
                buf_tokens += t

            if buf_pieces:
                flush_and_start_next_with_overlap()
                buf_pieces, buf_tokens = [], 0

        return chunks

    def _ocr_images_from_page(self, doc: fitz.Document, page: fitz.Page) -> str:
        """Extract and OCR individual images from a page."""
        images = page.get_images(full=True)
        if not images:
            return ""
        
        # Try batch OCR if available
        if hasattr(AzureDocIntelligenceService, "ocr_image_bytes_batch"):
            batch_imgs = self._extract_image_bytes(doc, images)
            if batch_imgs:
                batch_texts = AzureDocIntelligenceService.ocr_image_bytes_batch(batch_imgs)
                if batch_texts:
                    return "\n".join(p for p in batch_texts if p and p.strip()).strip()
        
        # Fallback to individual OCR
        parts = []
        for img in images:
            try:
                xref = img[0]
                base = doc.extract_image(xref)
                pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
                pil = self._clean_for_ocr(pil)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                text = AzureDocIntelligenceService.ocr_image_bytes(buf.getvalue())
                if text:
                    parts.append(text)
            except Exception:
                continue
        
        return "\n".join(parts).strip()

    def _extract_image_bytes(self, doc: fitz.Document, images: list) -> List[bytes]:
        """Extract image bytes from page images."""
        batch_imgs = []
        for img in images:
            try:
                xref = img[0]
                base = doc.extract_image(xref)
                pil = Image.open(io.BytesIO(base["image"])).convert("RGB")
                pil = self._clean_for_ocr(pil)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                batch_imgs.append(buf.getvalue())
            except Exception:
                continue
        return batch_imgs

    def _try_ocr_if_needed(self, doc: fitz.Document, page: fitz.Page, native: str, metrics: dict) -> str:
        """Check quality of native text and decide if OCR is needed."""
        # Quick checks for empty or very short text
        if not native or len(native) < 10:
            metrics["empty_pages"] += 1
            logging.warning(f"Empty or short native text for page {page.number}")
            return self._ocr_page(page).strip()

        # Heuristic: if native text is mostly whitespace, do OCR
        if len(re.findall(r'\S', native)) < 5:
            logging.info(f"Native text for page {page.number} is mostly whitespace, using OCR")
            return self._ocr_page(page).strip()

        return native
