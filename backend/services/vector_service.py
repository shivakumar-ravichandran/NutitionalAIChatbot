"""
Vector Embedding Service for semantic search and similarity matching
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime
import asyncio

# Import sentence-transformers when available
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Import FAISS when available
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


class VectorEmbeddingService:
    """Service for generating and managing vector embeddings"""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", index_dir: str = "./vector_index"
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)

        self.model = None
        self.index = None
        self.metadata = {}
        self.dimension = 384  # Default for all-MiniLM-L6-v2

        self._initialize_model()
        self._load_or_create_index()

    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Sentence Transformers not available. Using fallback embeddings."
            )
            return

        try:
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            self.model = None

    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Using fallback vector search.")
            return

        index_path = self.index_dir / "faiss_index.bin"
        metadata_path = self.index_dir / "metadata.json"

        try:
            if index_path.exists() and metadata_path.exists():
                # Load existing index
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"Loaded existing FAISS index with {self.index.ntotal} vectors"
                )
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(
                    self.dimension
                )  # Inner product for cosine similarity
                self.metadata = {
                    "created_at": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "dimension": self.dimension,
                    "documents": [],
                    "next_id": 0,
                }
                self._save_index()
                logger.info("Created new FAISS index")

        except Exception as e:
            logger.error(f"Error with FAISS index: {e}")
            self.index = None

    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        if not self.index:
            return

        try:
            index_path = self.index_dir / "faiss_index.bin"
            metadata_path = self.index_dir / "metadata.json"

            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            return self._fallback_embedding(text)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode([text], normalize_embeddings=True)[0]
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return self._fallback_embedding(text)

    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not self.model:
            return np.array([self._fallback_embedding(text) for text in texts])

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, lambda: self.model.encode(texts, normalize_embeddings=True)
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([self._fallback_embedding(text) for text in texts])

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding based on text characteristics"""
        # Create a simple hash-based embedding
        import hashlib

        # Use text characteristics to create a pseudo-embedding
        text_lower = text.lower()
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Create features based on text properties
        features = []

        # Length features
        features.extend(
            [
                len(text) / 1000,  # Normalized length
                len(text.split()) / 100,  # Normalized word count
            ]
        )

        # Character frequency features
        char_counts = {}
        for char in "abcdefghijklmnopqrstuvwxyz":
            char_counts[char] = text_lower.count(char) / len(text) if text else 0
        features.extend(list(char_counts.values()))

        # Hash-based features to reach target dimension
        hash_bytes = bytes.fromhex(text_hash)
        for i in range(self.dimension - len(features)):
            features.append((hash_bytes[i % len(hash_bytes)] / 255.0) * 2 - 1)

        return np.array(features[: self.dimension], dtype=np.float32)

    async def add_document(
        self, document_id: str, text: str, metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a document to the vector index"""
        try:
            # Generate embedding
            embedding = await self.generate_embedding(text)

            if self.index:
                # Add to FAISS index
                embedding_2d = embedding.reshape(1, -1)
                self.index.add(embedding_2d)

                # Store metadata
                doc_metadata = {
                    "id": document_id,
                    "text": text,
                    "added_at": datetime.now().isoformat(),
                    "index_position": self.metadata["next_id"],
                    **(metadata or {}),
                }

                self.metadata["documents"].append(doc_metadata)
                self.metadata["next_id"] += 1

                # Save updated index
                self._save_index()

                logger.info(f"Added document {document_id} to vector index")
                return True
            else:
                # Fallback storage
                logger.info(f"Added document {document_id} to fallback storage")
                return True

        except Exception as e:
            logger.error(f"Error adding document {document_id}: {e}")
            return False

    async def add_documents_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Add multiple documents to the vector index"""
        if not documents:
            return 0

        try:
            texts = [doc["text"] for doc in documents]
            embeddings = await self.generate_embeddings(texts)

            if self.index:
                # Add all embeddings to FAISS index
                self.index.add(embeddings)

                # Store metadata for all documents
                for i, doc in enumerate(documents):
                    doc_metadata = {
                        "id": doc.get("id", f"doc_{self.metadata['next_id']}"),
                        "text": doc["text"],
                        "added_at": datetime.now().isoformat(),
                        "index_position": self.metadata["next_id"],
                        **doc.get("metadata", {}),
                    }

                    self.metadata["documents"].append(doc_metadata)
                    self.metadata["next_id"] += 1

                # Save updated index
                self._save_index()

                logger.info(f"Added {len(documents)} documents to vector index")
                return len(documents)
            else:
                logger.info(f"Added {len(documents)} documents to fallback storage")
                return len(documents)

        except Exception as e:
            logger.error(f"Error adding documents batch: {e}")
            return 0

    async def search_similar(
        self, query: str, k: int = 10, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)

            if self.index and self.index.ntotal > 0:
                # Search using FAISS
                query_2d = query_embedding.reshape(1, -1)
                scores, indices = self.index.search(query_2d, min(k, self.index.ntotal))

                # Prepare results
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if (
                        idx >= 0 and score >= threshold
                    ):  # Valid index and above threshold
                        doc_metadata = self.metadata["documents"][idx]
                        results.append(
                            {
                                "id": doc_metadata["id"],
                                "text": doc_metadata["text"],
                                "score": float(score),
                                "metadata": {
                                    k: v
                                    for k, v in doc_metadata.items()
                                    if k not in ["id", "text"]
                                },
                                "rank": i + 1,
                            }
                        )

                return results
            else:
                # Fallback similarity search
                return self._fallback_search(query, k, threshold)

        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []

    def _fallback_search(
        self, query: str, k: int, threshold: float
    ) -> List[Dict[str, Any]]:
        """Simple fallback search using text matching"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Simple keyword matching
        for doc in self.metadata.get("documents", []):
            text_lower = doc["text"].lower()
            text_words = set(text_lower.split())

            # Calculate simple similarity
            intersection = len(query_words.intersection(text_words))
            union = len(query_words.union(text_words))
            jaccard_similarity = intersection / union if union > 0 else 0

            # Boost score if query is substring of text
            if query_lower in text_lower:
                jaccard_similarity += 0.2

            if jaccard_similarity >= threshold:
                results.append(
                    {
                        "id": doc["id"],
                        "text": doc["text"],
                        "score": jaccard_similarity,
                        "metadata": {
                            k: v for k, v in doc.items() if k not in ["id", "text"]
                        },
                        "rank": 0,
                    }
                )

        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, result in enumerate(results[:k]):
            result["rank"] = i + 1

        return results[:k]

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID"""
        for doc in self.metadata.get("documents", []):
            if doc["id"] == document_id:
                return {
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": {
                        k: v for k, v in doc.items() if k not in ["id", "text"]
                    },
                }
        return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector index"""
        try:
            # Find document in metadata
            doc_to_remove = None
            doc_index = None

            for i, doc in enumerate(self.metadata.get("documents", [])):
                if doc["id"] == document_id:
                    doc_to_remove = doc
                    doc_index = i
                    break

            if not doc_to_remove:
                logger.warning(f"Document {document_id} not found")
                return False

            # Remove from metadata
            self.metadata["documents"].pop(doc_index)

            # Note: FAISS doesn't support individual deletion efficiently
            # In production, consider rebuilding index periodically
            logger.warning(
                "Document removed from metadata. Consider rebuilding FAISS index."
            )

            # Save updated metadata
            self._save_index()

            logger.info(f"Document {document_id} marked for deletion")
            return True

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False

    async def rebuild_index(self) -> bool:
        """Rebuild the entire FAISS index"""
        try:
            if not self.metadata.get("documents"):
                logger.info("No documents to rebuild index")
                return True

            # Extract texts from all documents
            texts = [doc["text"] for doc in self.metadata["documents"]]

            # Generate embeddings for all documents
            embeddings = await self.generate_embeddings(texts)

            if FAISS_AVAILABLE:
                # Create new index
                new_index = faiss.IndexFlatIP(self.dimension)
                new_index.add(embeddings)

                # Replace old index
                self.index = new_index

                # Update index positions in metadata
                for i, doc in enumerate(self.metadata["documents"]):
                    doc["index_position"] = i

                # Save new index
                self._save_index()

                logger.info(f"Rebuilt FAISS index with {len(texts)} documents")
                return True
            else:
                logger.info("FAISS not available, metadata updated")
                return True

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        return {
            "total_documents": len(self.metadata.get("documents", [])),
            "index_dimension": self.dimension,
            "model_name": self.model_name,
            "faiss_available": FAISS_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "index_size": self.index.ntotal if self.index else 0,
            "created_at": self.metadata.get("created_at"),
            "last_updated": datetime.now().isoformat(),
        }


# Global vector service instance
vector_service = None


def get_vector_service() -> VectorEmbeddingService:
    """Get or create vector embedding service instance"""
    global vector_service
    if vector_service is None:
        vector_service = VectorEmbeddingService()
    return vector_service


async def initialize_vector_service(
    model_name: str = "all-MiniLM-L6-v2", index_dir: str = "./vector_index"
) -> VectorEmbeddingService:
    """Initialize vector embedding service on startup"""
    global vector_service
    try:
        vector_service = VectorEmbeddingService(model_name, index_dir)
        logger.info("Vector embedding service initialized")
        return vector_service
    except Exception as e:
        logger.error(f"Failed to initialize vector service: {e}")
        # Return service anyway for fallback functionality
        vector_service = VectorEmbeddingService(model_name, index_dir)
        return vector_service
