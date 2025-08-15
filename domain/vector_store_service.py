import logging
import time
from pathlib import Path
from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from domain.email_repository import EmailRepository

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Optimized service for managing vector embeddings of emails using Qdrant vector database.
    """

    def __init__(
        self,
        collection_name: str = "emails",
        vector_store_path: str = "./data/vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.collection_name = collection_name
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure vector store directory exists
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant client
        self.qdrant_path = self.vector_store_path / "qdrant_db"
        self.client = QdrantClient(path=str(self.qdrant_path))

        # Initialize embedding model (cached)
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

        # Initialize text splitter (cached)
        self.text_splitter = SentenceSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Cache for vector store and index
        self._vector_store: QdrantVectorStore | None = None
        self._index: VectorStoreIndex | None = None
        self._query_engine: Any | None = None

        # Initialize the vector store
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize the vector store collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                # Create new collection with correct dimensions for the embedding model
                vector_size = 384  # For sentence-transformers/all-MiniLM-L6-v2
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def _get_vector_store(self) -> QdrantVectorStore:
        """Get cached vector store instance."""
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client, collection_name=self.collection_name
            )
        return self._vector_store

    def _get_index(self) -> VectorStoreIndex:
        """Get cached index instance."""
        if self._index is None:
            vector_store = self._get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self.embed_model,
            )
        return self._index

    def _get_query_engine(self, top_k: int = 5) -> Any:
        """Get cached query engine instance."""
        if self._query_engine is None:
            index = self._get_index()
            self._query_engine = index.as_query_engine(
                similarity_top_k=top_k, response_mode="compact"
            )
        return self._query_engine

    def _create_documents_from_emails(
        self, emails: list[dict[str, Any]]
    ) -> list[Document]:
        """Convert emails to LlamaIndex Documents."""
        documents = []

        for email in emails:
            # Combine relevant fields for embedding
            content_parts = []

            if email.get("subject"):
                content_parts.append(f"Subject: {email['subject']}")

            if email.get("sender_name") or email.get("sender_email"):
                sender_info = f"From: {email.get('sender_name', '')} <{email.get('sender_email', '')}>"
                content_parts.append(sender_info)

            if email.get("body"):
                content_parts.append(f"Content: {email['body']}")

            if email.get("summary"):
                content_parts.append(f"Summary: {email['summary']}")

            # Add metadata fields
            metadata = {
                "email_id": email.get("id"),
                "subject": email.get("subject", ""),
                "sender_name": email.get("sender_name", ""),
                "sender_email": email.get("sender_email", ""),
                "email_type": email.get("email_type", ""),
                "category": email.get("category", ""),
                "vendor": email.get("vendor", ""),
                "amount": email.get("amount", ""),
                "source": email.get("source", ""),
                "destination": email.get("destination", ""),
                "date": email.get("date", ""),
            }

            # Create document
            content = "\n".join(content_parts)
            if content.strip():
                doc = Document(text=content, metadata=metadata)
                documents.append(doc)

        return documents

    def build_index_from_emails(
        self,
        email_repository: EmailRepository,
        start: int = 0,
        limit: int | None = None,
        batch_size: int = 100,
        clear_existing: bool = False,
    ) -> int:
        """
        Build vector index from emails in the database with optimized batch processing.
        """
        if clear_existing:
            self.clear_index()

        total_processed = 0
        current_offset = start

        # Get vector store and storage context once
        vector_store = self._get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        while True:
            # Calculate batch limit
            batch_limit = batch_size
            if limit and (current_offset + batch_size) > (start + limit):
                batch_limit = (start + limit) - current_offset

            if batch_limit <= 0:
                break

            # Fetch batch of emails
            emails = email_repository.get_emails(
                start=current_offset, limit=batch_limit
            )

            if not emails:
                break

            # Convert to documents
            documents = self._create_documents_from_emails(emails)

            if documents:
                # Process documents with text splitter
                nodes = self.text_splitter.get_nodes_from_documents(documents)

                # Create index for this batch and add to vector store
                batch_index = VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    embed_model=self.embed_model,
                )

                total_processed += len(emails)
                logger.info(
                    f"Processed batch: {len(emails)} emails (Total: {total_processed})"
                )

            current_offset += batch_limit

            # Check if we've reached the limit
            if limit and current_offset >= (start + limit):
                break

        # Clear cache to force refresh on next search
        self._clear_cache()

        logger.info(
            f"Vector index built successfully. Total emails processed: {total_processed}"
        )
        return total_processed

    def search_similar_emails(
        self, query: str, top_k: int = 5, similarity_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Optimized search for similar emails using vector similarity.
        """
        try:
            start_time = time.time()

            # Get cached query engine
            query_engine = self._get_query_engine(top_k=top_k)

            # Execute query
            response = query_engine.query(query)

            # Format results
            results = []
            for node in response.source_nodes:
                if node.score and node.score >= similarity_threshold:
                    results.append(
                        {
                            "score": node.score,
                            "content": node.text,
                            "metadata": node.metadata,
                        }
                    )

            search_time = time.time() - start_time
            logger.info(
                f"Vector search completed in {search_time:.3f}s for query: {query[:50]}..."
            )

            return results

        except Exception as e:
            logger.error(f"Error searching similar emails: {e}")
            return []

    def _clear_cache(self):
        """Clear cached instances."""
        self._vector_store = None
        self._index = None
        self._query_engine = None

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the vector index."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vector_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status.value,
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

    def clear_index(self):
        """Clear all vectors from the index."""
        try:
            self.client.delete_collection(self.collection_name)
            self._clear_cache()
            self._init_vector_store()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
