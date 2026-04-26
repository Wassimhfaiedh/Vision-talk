"""
Vector memory system for storing and searching video captions.
Semantic search with cosine similarity scoring and reranking.
No frame caching - only stores captions, timestamps, and metadata.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import numpy as np
from math import exp

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Sentence Transformers not available")


class VectorMemory:
    """ChromaDB vector database manager."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.embedder = None
        self.cross_encoder = None
        self.client = None
        self.collection = None
        self.initialize()

    def initialize(self, collection_name: Optional[str] = None):
        """Initialize ChromaDB client and load embedding model."""
        if collection_name is None:
            collection_name = self.config.COLLECTION_NAME

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.config.CHROMA_DB_PATH),
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )

            if SENTENCE_TRANSFORMERS_AVAILABLE:
                print(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
                self.embedder = SentenceTransformer(self.config.EMBEDDING_MODEL)
                
                try:
                    print(f"Loading cross-encoder: {self.config.RERANKING_MODEL}")
                    self.cross_encoder = CrossEncoder(self.config.RERANKING_MODEL)
                except Exception as e:
                    print(f"Could not load cross-encoder: {e}")
                    self.cross_encoder = None
            else:
                raise ImportError("Sentence Transformers is required for this system")

            try:
                self.collection = self.client.get_collection(name=collection_name)
                print(f"Loaded collection: {collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "Vision-Talk captions database"}
                )
                print(f"Created collection: {collection_name}")

            return True

        except Exception as e:
            print(f"Database initialization error: {e}")
            return False

    # ============================================================================
    # STORAGE
    # ============================================================================

    def store_caption_realtime(self, caption_data: Dict, video_source: str,
                                video_path: str = None) -> bool:
        """Store a single caption in the vector database."""
        if not caption_data:
            print("⚠️ No caption data to store")
            return False

        try:
            caption_id = str(uuid.uuid4())
            caption_text = caption_data.get('caption', '')
            timestamp_type = caption_data.get('timestamp_type', 'elapsed')
            timestamp_value = float(caption_data.get('timestamp_value', 0))
            timestamp_display = caption_data.get('timestamp_display', '00:00:00')

            metadata = {
                'timestamp_type': timestamp_type,
                'timestamp_value': timestamp_value,
                'timestamp_display': timestamp_display,
                'video_source': video_source,
                'video_path': video_path or '',
                'frame_count': int(caption_data.get('frame_count', 0)),
                'added_date': datetime.now().isoformat(),
                'caption': caption_text  
            }
            embedding = self.embedder.encode([caption_text]).tolist()
            self.collection.add(
                embeddings=embedding,
                documents=[caption_text],
                metadatas=[metadata],
                ids=[caption_id]
            )
            print(f"✅ Caption stored at {timestamp_display}")

            return True

        except Exception as e:
            print(f"Real-time storage error: {e}")
            return False

    # ============================================================================
    # SEARCH
    # ============================================================================

    def search_memory(self, query: str, n_results: int = 10,
                      filter_source: str = None) -> List[Dict]:
        """Semantic search in database using cosine similarity and optional reranking."""
        try:
            total = self.collection.count()
            if total == 0:
                print("📭 Database empty")
                return []

            # Get more results initially if reranking is enabled
            if self.cross_encoder:
                initial_n = min(n_results * self.config.RERANKING_MULTIPLIER, total)
            else:
                initial_n = min(n_results, total)

            where_filter = None
            if filter_source:
                where_filter = {"video_source": filter_source}

            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Query the database
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_n,
                where=where_filter,
                include=["metadatas", "distances", "documents", "embeddings"]
            )

            formatted_results = []

            if results['ids'] and results['ids'][0]:
                query_emb_np = np.array(query_embedding)
                
                for i in range(len(results['ids'][0])):
                    metadata = results['metadatas'][0][i]
                    caption = results['documents'][0][i]
                    
                    # Calculate cosine similarity
                    result_embedding = np.array(results['embeddings'][0][i])
                    dot_product = np.dot(query_emb_np, result_embedding)
                    norm_query = np.linalg.norm(query_emb_np)
                    norm_result = np.linalg.norm(result_embedding)
                    
                    if norm_query > 0 and norm_result > 0:
                        cosine_sim = dot_product / (norm_query * norm_result)
                        similarity = round(float(cosine_sim), 3)
                    else:
                        similarity = 0.0

                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'caption': caption,
                        'timestamp_type': metadata.get('timestamp_type', 'elapsed'),
                        'timestamp': float(metadata.get('timestamp_value', 0)),
                        'timestamp_display': metadata.get('timestamp_display', '00:00:00'),
                        'video_source': metadata['video_source'],
                        'video_path': metadata.get('video_path', ''),
                        'frame_count': int(metadata.get('frame_count', 0)),
                        'score': similarity
                    })

                # Apply reranking if cross-encoder is available
                if self.cross_encoder and len(formatted_results) > 1:
                    formatted_results = self._rerank_results(query, formatted_results)
                else:
                    formatted_results.sort(key=lambda x: x['score'], reverse=True)
                
                # Return only top n_results
                formatted_results = formatted_results[:n_results]

            print(f"Found {len(formatted_results)} results for: '{query}'")
            return formatted_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank search results using cross-encoder for better relevance."""
        try:
            pairs = [(query, result['caption']) for result in results]
            rerank_scores = self.cross_encoder.predict(pairs)
            
            for i, score in enumerate(rerank_scores):
                if hasattr(score, '__float__'):
                    normalized_score = 1.0 / (1.0 + exp(-float(score)))
                else:
                    normalized_score = float(score)
                normalized_score = max(0.0, min(1.0, normalized_score))
                results[i]['score'] = round(normalized_score, 3)
            
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
            
        except Exception as e:
            print(f"Reranking error: {e}")
            results.sort(key=lambda x: x['score'], reverse=True)
            return results

    # ============================================================================
    # Q&A PREPARATION
    # ============================================================================

    def prepare_for_qa(self, query: str, n_results: int = 10) -> Dict:
        """Prepare context for Q&A with Gemini."""
        results = self.search_memory(query, n_results=n_results)

        if not results:
            return {
                'results': [],
                'context': "No relevant video segments found in the database.",
                'result_count': 0
            }
        
        context_parts = []
        for i, result in enumerate(results, 1):
            timestamp_display = result.get('timestamp_display', self._format_timestamp(result['timestamp']))
            context_parts.append(
                f"[Segment {i} - {timestamp_display} from {result['video_source']}]\n"
                f"Caption: {result['caption']}\n"
                f"Relevance Score: {result['score']:.3f}\n"
            )

        return {
            'results': results,
            'context': "\n".join(context_parts),
            'result_count': len(results)
        }

    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    # ============================================================================
    # DATABASE MANAGEMENT
    # ============================================================================

    def get_video_sources(self) -> List[str]:
        """Get all unique video sources."""
        try:
            all_items = self.collection.get()
            if not all_items['metadatas']:
                return []

            sources = set()
            for meta in all_items['metadatas']:
                if 'video_source' in meta:
                    sources.add(meta['video_source'])
            return sorted(list(sources))

        except Exception as e:
            print(f"Error getting sources: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            sources = self.get_video_sources()

            return {
                'total_items': count,
                'video_sources': sources,
                'source_count': len(sources),
                'has_embeddings': self.embedder is not None,
                'has_reranking': self.cross_encoder is not None
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {'total_items': 0, 'video_sources': [], 'source_count': 0}

    def reset_database(self) -> bool:
        """Reset the entire database."""
        try:
            self.client.reset()
            print("🗑️ Database reset")
            self.initialize()
            return True
        except Exception as e:
            print(f"Reset error: {e}")
            return False