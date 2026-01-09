from datetime import datetime
import logging
from typing import Dict, Iterable, List, Optional, Tuple
from config_public import (
    ZILLIZ_URI, ZILLIZ_API_KEY, ZILLIZ_DB_NAME
)
from vector_sync.data_models import SharePointFileDescriptor
from API_SharePoint import SharePointContentAndMetadataService
from vector_sync.proposal_ingest import ProposalIngestor
from vector_sync.document_parsing_and_chunking_service import DocumentParsingAndChunkingService
from API_Azure import AzureOpenAIPlatformService
from API_Zilliz import ZillizVectorStorageService
from utils.retry import retry
import time

class VectorSyncOrchestrator:

    def __init__(self):
        self._sp = SharePointContentAndMetadataService()
        self.proposal_ingestor = ProposalIngestor()
        self._parser = DocumentParsingAndChunkingService()
        self._openai = AzureOpenAIPlatformService()
        self._zilliz = ZillizVectorStorageService(ZILLIZ_URI, ZILLIZ_API_KEY, ZILLIZ_DB_NAME)


    # Delete vectors from Zilliz
    def delete_file_by_sharepoint_id(self, collection_name: str, sharepoint_file_id: str) -> None:
        self._zilliz.zilliz_delete_all_chunks_for_document(collection_name, sharepoint_file_id)


    @retry(retries=3, base_backoff=2.0, jitter=0.3)
    def zilliz_insert_with_retry(self, collection_name: str, rows: List[dict]):
        self._zilliz.zilliz_insert_chunk_rows(collection_name, rows)

    # Get sharepoint ID
    def _doc_id(self, sp_file: SharePointFileDescriptor, fallback: str) -> str:
        return sp_file.listitem_id or fallback

    # Single Upsert
    def upsert_single_file_from_sharepoint_url(self, collection_name: str, full_sharepoint_url: str) -> None:
        from datetime import datetime
        start_time = datetime.now()

        # Get SharePoint file descriptor
        sp_file = self._sp.sharepoint_get_descriptor_by_url(full_sharepoint_url)
        doc_id = self._doc_id(sp_file, full_sharepoint_url)

        # Log file name and folder path
        file_name = getattr(sp_file, 'file_name', None)
        folder_path = getattr(sp_file, 'parent_reference_path', None) or getattr(sp_file, 'folder_path', None)
        logging.info(f"Processing file: name='{file_name}', folder='{folder_path}', url='{full_sharepoint_url}'")
        print(f"Processing file: name='{file_name}', folder='{folder_path}', url='{full_sharepoint_url}'")

        # Download and chunk the file
        raw_bytes = self._sp.sharepoint_download_bytes_by_url(full_sharepoint_url)
        page_paragraphs = self._parser.extract_paragraphs_from_bytes(sp_file.file_name, raw_bytes)
        if collection_name == "ProposalDocs":
            full_text = "\n\n".join([text for _, text in page_paragraphs])
            chunk_tuples, _ = self.proposal_ingestor.process_single_proposal(file_name, full_text)
            # No flattened_metadata needed - everything is in chunk_text
        else:
            chunk_tuples = self._parser.build_chunk_tuples(page_paragraphs)

        logging.info(f"Document chunked: doc_id={doc_id} -> {len(chunk_tuples)} chunks to embed.")
        print(f"Document chunked: doc_id={doc_id} -> {len(chunk_tuples)} chunks to embed.")

        # Build custom metadata
        lib_meta = self._sp.sharepoint_build_metadata_for_file(sp_file, collection_name)

        # Remove all existing vector chunks for this document from the Zilliz collection
        # This prevents duplicate embeddings and ensures only the latest data is stored
        self._zilliz.zilliz_delete_all_chunks_for_document(collection_name, doc_id)

        # --- Batching for embeddings and Zilliz inserts ---
        EMBEDDING_BATCH_SIZE = 15
        ZILLIZ_BATCH_SIZE = 30
        rows: List[dict] = []
        total_chunks = len(chunk_tuples)
        for i in range(0, total_chunks, EMBEDDING_BATCH_SIZE):
            batch = chunk_tuples[i:i+EMBEDDING_BATCH_SIZE]
            texts = []
            meta = []
            for page_num, chunk_idx, chunk_text in batch:
                if len(chunk_text) > 8000:
                    logging.warning(f"Oversized chunk {chunk_idx} in {sp_file.file_name}, length={len(chunk_text)}")
                texts.append(chunk_text)
                meta.append((page_num, chunk_idx, chunk_text))
            embeddings = self._openai.generate_embeddings_bulk(texts)
            time.sleep(0.1)
            for (page_num, chunk_idx, chunk_text), emb in zip(meta, embeddings):
                row = {
                    "sp_id": doc_id,
                    "file_name": sp_file.file_name,
                    "path": full_sharepoint_url,
                    "listitem_id": sp_file.listitem_id,
                    "modified": sp_file.last_modified_iso,
                    "chunk": chunk_idx,
                    "page_num": float(page_num) if page_num is not None else None,
                    "sharepoint_link": f"{sp_file.additional_graph_fields.get('webUrl')}#page={page_num}" if sp_file.additional_graph_fields.get("webUrl") else None,
                    "vector": emb,
                    "chunk_text": chunk_text,  # This now contains summary + metadata
                    **lib_meta  # Only merge library metadata
                }
                rows.append(row)
                # Insert to Zilliz in batches
                if len(rows) >= ZILLIZ_BATCH_SIZE:
                    self.zilliz_insert_with_retry(collection_name, rows)
                    logging.info(f"Inserted {len(rows)} rows to Zilliz (intermediate batch) for doc_id={doc_id}")
                    rows = []
        # Insert any remaining rows
        if rows:
            self.zilliz_insert_with_retry(collection_name, rows)
            logging.info(f"Inserted {len(rows)} rows to Zilliz (final batch) for doc_id={doc_id}")
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Upserted {total_chunks} chunks for doc_id={doc_id} in collection={collection_name}. Duration: {duration}")
        print(f"Upserted {total_chunks} chunks for doc_id={doc_id} in collection={collection_name}. Duration: {duration}")